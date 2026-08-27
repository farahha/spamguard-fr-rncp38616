"""Expérience Machine Learning classique reproductible de SpamGuard-FR."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from spamguard.config import PROJECT_ROOT, load_config
from spamguard.data import (
    deduplicate_french_text,
    load_frozen_split,
    load_processed_data,
    save_deduplicated_data,
)
from spamguard.evaluation import business_confusion_counts
from spamguard.persistence import save_markdown


MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "random_forest": "Random Forest",
}


def build_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def build_ml_pipelines(random_state: int = 42) -> dict[str, Pipeline]:
    """Construit trois pipelines indépendants avec le même TF-IDF."""
    estimators = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            solver="liblinear",
        ),
        "linear_svm": LinearSVC(class_weight="balanced", random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    return {
        name: Pipeline([("tfidf", build_tfidf_vectorizer()), ("classifier", estimator)])
        for name, estimator in estimators.items()
    }


def _time_inference(pipeline: Pipeline, texts: pd.Series, repeats: int = 20) -> float:
    started = time.perf_counter()
    for _ in range(repeats):
        pipeline.predict(texts)
    elapsed = time.perf_counter() - started
    return elapsed / (repeats * len(texts)) * 1000


def _save_confusion_figure(y_true, y_pred, model_key: str, figures_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=["ham", "spam"],
        display_labels=["ham", "spam"],
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title(f"Matrice de confusion — {MODEL_NAMES[model_key]}")
    figure.tight_layout()
    figure.savefig(figures_dir / f"confusion_matrix_{model_key}.png", dpi=150)
    plt.close(figure)


def _save_comparison_figure(metrics: pd.DataFrame, figures_dir: Path) -> None:
    columns = ["accuracy", "precision_spam", "recall_spam", "f1_spam", "f1_macro"]
    plot_data = metrics.set_index("model")[columns]
    axis = plot_data.plot.bar(figsize=(11, 5), ylim=(0, 1), rot=0, width=0.8)
    axis.set(title="Comparaison des modèles sur le même jeu de test", ylabel="Score", xlabel="")
    axis.legend(loc="lower right", ncol=2)
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(figures_dir / "ml_model_comparison.png", dpi=150)
    plt.close(axis.figure)


def _error_examples(test: pd.DataFrame, predictions, limit: int = 5) -> dict:
    errors = test[["id", "text_fr", "text_en", "label"]].copy()
    errors["prediction"] = predictions
    return {
        "false_positives": errors[(errors["label"] == "ham") & (errors["prediction"] == "spam")].head(limit),
        "false_negatives": errors[(errors["label"] == "spam") & (errors["prediction"] == "ham")].head(limit),
    }


def _comparison_report(
    before: pd.DataFrame,
    deduplicated: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    metrics: pd.DataFrame,
    selected_key: str,
    error_examples: dict,
) -> str:
    selected_name = MODEL_NAMES[selected_key]
    table = metrics.to_markdown(index=False, floatfmt=".6f")
    selected_errors = error_examples[selected_key]
    fp_texts = selected_errors["false_positives"]["text_fr"].tolist()
    fn_texts = selected_errors["false_negatives"]["text_fr"].tolist()
    confusion_rows = [
        (
            f"| {row['model']} | {len(test[test['label'] == 'ham']) - int(row['false_positives'])} "
            f"| {int(row['false_positives'])} | {int(row['false_negatives'])} "
            f"| {len(test[test['label'] == 'spam']) - int(row['false_negatives'])} |"
        )
        for _, row in metrics.iterrows()
    ]
    lines = [
        "# Comparaison des modèles Machine Learning",
        "",
        "Les trois pipelines rechargent les identifiants du split historique et ajustent leur TF-IDF uniquement sur le train.",
        "",
        "## Données et split",
        "",
        f"- Avant déduplication : {len(before)} messages",
        f"- Après déduplication stricte sur `text_fr` : {len(deduplicated)} messages",
        f"- Lignes supprimées : {len(before) - len(deduplicated)}",
        f"- Train : {len(train)} messages ; test : {len(test)} messages",
        f"- Train — ham : {(train['label'] == 'ham').sum()}, spam : {(train['label'] == 'spam').sum()}",
        f"- Test — ham : {(test['label'] == 'ham').sum()}, spam : {(test['label'] == 'spam').sum()}",
        "- Intersection exacte des textes entre train et test : 0",
        "",
        "L'audit initial trouvait 415 doublons du triplet complet. La règle expérimentale plus large, fondée uniquement sur `text_fr`, retire davantage de lignes quand plusieurs originaux anglais aboutissent à la même traduction française.",
        "",
        "## Résultats sur le test uniquement",
        "",
        table,
        "",
        "TP/TN/FP/FN utilisent `spam` comme classe positive. Un faux positif est un SMS légitime écarté ; un faux négatif est un spam qui atteint l'utilisateur.",
        "",
        "| Modèle | TN | FP | FN | TP |",
        "|---|---:|---:|---:|---:|",
        *confusion_rows,
        "",
        "## Décision",
        "",
        f"Le modèle retenu est **{selected_name}**, sélectionné sur le F1 spam, avec contrôle conjoint de la precision, du recall, des faux positifs, des faux négatifs et des temps d'exécution.",
        "",
        "La métrique principale retenue est le **F1 spam**. Elle équilibre la protection contre les spams non détectés et la préservation des messages légitimes. Le recall spam et le nombre de faux positifs restent des garde-fous obligatoires.",
        "",
        "## Exemples d'erreurs du modèle retenu",
        "",
        "### Faux positifs",
        *([f"- {text}" for text in fp_texts] or ["- Aucun sur ce split."]),
        "",
        "### Faux négatifs",
        *([f"- {text}" for text in fn_texts] or ["- Aucun sur ce split."]),
        "",
        "Ces messages sont présentés pour inspection qualitative seulement. Une explication n'est retenue que si elle est directement visible dans le texte (brièveté, chiffres, URL, ambiguïté ou traduction maladroite).",
        "",
        "## Limites",
        "",
        "- Les textes français sont des traductions automatiques et non un corpus natif français.",
        "- Le jeu de test n'est utilisé qu'une fois pour cette comparaison, mais une future optimisation exigera une validation distincte ou une validation croisée sur le train.",
        "- Les paramètres TF-IDF et modèles sont des configurations raisonnables, pas des optimums démontrés.",
        "- Les temps dépendent de cette machine et de la charge du système.",
    ]
    return "\n".join(lines) + "\n"


def _metric_adr(metrics: pd.DataFrame) -> str:
    table = metrics[["model", "accuracy", "precision_spam", "recall_spam", "f1_spam", "f1_macro", "false_positives", "false_negatives"]].to_markdown(index=False, floatfmt=".4f")
    return f"""# ADR-001 — Sélection de la métrique prioritaire

- Statut : accepté
- Date : 2026-08-24

## Contexte

Avant déduplication, le corpus comporte 86,59 % de ham et 13,41 % de spam. Une prédiction systématique de la classe majoritaire obtiendrait donc une accuracy apparemment élevée sans répondre au besoin de détection. Un faux positif masque un message légitime ; un faux négatif laisse parvenir un spam potentiellement frauduleux.

## Résultats observés sur le test

{table}

## Décision

La métrique principale du projet est le **F1-score de la classe spam**. Elle impose un équilibre explicite entre precision spam et recall spam, contrairement à l'accuracy, et reste centrée sur la classe métier d'intérêt.

Le F1 spam ne sera jamais lu isolément : le **recall spam** et le **nombre de faux positifs** sont des garde-fous obligatoires. Cette lecture conjointe empêche qu'un bon F1 masque soit trop de spams non détectés, soit une dégradation excessive des messages légitimes.

## Options écartées comme métrique unique

- Accuracy : trop influencée par les ham majoritaires.
- Precision spam : limiterait les faux positifs, mais pourrait tolérer trop de faux négatifs.
- Recall spam : protège fortement contre les spams, mais peut augmenter les SMS légitimes bloqués.
- F1 macro : utile pour comparer l'équilibre global, mais moins directement lié à la classe spam.

## Conséquences

Les comparaisons futures classeront les candidats sur le F1 spam puis contrôleront recall spam, faux positifs, F1 macro, simplicité et coût d'exécution. Toute modification du coût métier des erreurs devra rouvrir cette ADR.
"""


def _model_adr(metrics: pd.DataFrame, selected_key: str) -> str:
    selected_name = MODEL_NAMES[selected_key]
    row = metrics.loc[metrics["model"] == selected_name].iloc[0]
    return f"""# ADR-002 — Sélection du modèle Machine Learning classique

- Statut : accepté
- Date : 2026-08-24
- Niveau de confiance : modéré

## Contexte et critères

Trois pipelines TF-IDF ont été évalués sur le même test historique : Logistic Regression, Linear SVM et Random Forest. Les critères sont le F1 spam principal, la precision et le recall spam, les faux positifs/négatifs, la simplicité, les temps d'entraînement et d'inférence, et l'interprétabilité.

## Avantages et inconvénients

- **Logistic Regression** : linéaire, coefficients inspectables et compromis de classes réglable ; peut être moins performante qu'un SVM sur un espace sparse très dimensionnel.
- **Linear SVM** : particulièrement adapté aux vecteurs sparse de grande dimension et souvent efficace en classification de texte ; scores non probabilistes et interprétation moins immédiate qu'une régression logistique.
- **Random Forest** : représente la famille des ensembles d'arbres étudiée dans la formation, capture des interactions non linéaires et fournit des importances ; il est plus coûteux, moins compact et n'est pas naturellement optimal pour un TF-IDF sparse de grande dimension.

Note historique : une première exploration incluait Multinomial Naive Bayes. Elle a été retirée de la comparaison finale parce que cet algorithme ne fait pas partie du programme pédagogique suivi.

## Décision

Le pipeline retenu est **{selected_name}**. Sur cette exécution, il obtient un F1 spam de **{row['f1_spam']:.4f}**, une precision spam de **{row['precision_spam']:.4f}**, un recall spam de **{row['recall_spam']:.4f}**, avec **{int(row['false_positives'])} faux positifs** et **{int(row['false_negatives'])} faux négatifs**.

Son temps d'entraînement mesuré est de **{row['training_time_seconds']:.6f} s** et son inférence moyenne de **{row['inference_time_ms_mean']:.6f} ms par message**. La sélection suit le F1 spam puis vérifie les garde-fous métier et la sobriété opérationnelle.

## Conséquences

Seul ce pipeline TF-IDF complet est sauvegardé dans `models/ml_spam_classifier.joblib`. Les deux autres restent reproductibles via le code mais ne sont pas persistés. Le niveau de confiance est modéré car la comparaison repose sur un seul split et sur un corpus français traduit automatiquement ; la décision devra être réévaluée avec validation croisée et données françaises natives.
"""


def run_ml_experiment() -> dict:
    config = load_config()
    random_state = int(config["ml"]["random_state"])
    before = load_processed_data()
    deduplicated = deduplicate_french_text(before)
    save_deduplicated_data(deduplicated)
    train, test = load_frozen_split(deduplicated)
    if set(train["text_fr"]).intersection(test["text_fr"]):
        raise RuntimeError("Fuite détectée : texte français commun au train et au test.")

    figures_dir = PROJECT_ROOT / config["paths"]["figures"]
    metrics_dir = PROJECT_ROOT / config["paths"]["metrics"]
    reports_dir = PROJECT_ROOT / config["paths"]["reports"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pipelines = build_ml_pipelines(random_state)
    rows = []
    reports = {}
    errors = {}
    predictions_by_model = {}
    for key, pipeline in pipelines.items():
        started = time.perf_counter()
        pipeline.fit(train["text_fr"], train["label"])
        training_seconds = time.perf_counter() - started
        predictions = pipeline.predict(test["text_fr"])
        inference_ms = _time_inference(pipeline, test["text_fr"])
        counts = business_confusion_counts(test["label"], predictions)
        rows.append({
            "model": MODEL_NAMES[key],
            "accuracy": accuracy_score(test["label"], predictions),
            "precision_spam": precision_score(test["label"], predictions, pos_label="spam", zero_division=0),
            "recall_spam": recall_score(test["label"], predictions, pos_label="spam", zero_division=0),
            "f1_spam": f1_score(test["label"], predictions, pos_label="spam", zero_division=0),
            "f1_macro": f1_score(test["label"], predictions, average="macro", zero_division=0),
            "false_positives": counts["false_positives"],
            "false_negatives": counts["false_negatives"],
            "training_time_seconds": training_seconds,
            "inference_time_ms_mean": inference_ms,
        })
        reports[key] = classification_report(test["label"], predictions, digits=4, zero_division=0)
        errors[key] = _error_examples(test, predictions)
        predictions_by_model[key] = predictions
        _save_confusion_figure(test["label"], predictions, key, figures_dir)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_dir / "ml_model_comparison.csv", index=False)
    _save_comparison_figure(metrics, figures_dir)
    selected_key = next(
        key for key, name in MODEL_NAMES.items()
        if name == metrics.sort_values(["f1_spam", "recall_spam"], ascending=False).iloc[0]["model"]
    )
    model_path = PROJECT_ROOT / config["paths"]["ml_model"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipelines[selected_key], model_path)
    reloaded = joblib.load(model_path)
    sample_texts = test["text_fr"].head(5)
    if not (reloaded.predict(sample_texts) == pipelines[selected_key].predict(sample_texts)).all():
        raise RuntimeError("La vérification du modèle rechargé a échoué.")

    save_markdown(
        _comparison_report(before, deduplicated, train, test, metrics, selected_key, errors),
        reports_dir / "ml_model_comparison.md",
    )
    save_markdown(_metric_adr(metrics), PROJECT_ROOT / "docs/adr/ADR-001-metric-selection.md")
    save_markdown(_model_adr(metrics, selected_key), PROJECT_ROOT / "docs/adr/ADR-002-ml-model-selection.md")
    return {
        "before": before,
        "deduplicated": deduplicated,
        "train": train,
        "test": test,
        "metrics": metrics,
        "classification_reports": reports,
        "errors": errors,
        "pipelines": pipelines,
        "predictions": predictions_by_model,
        "selected_key": selected_key,
        "selected_model": MODEL_NAMES[selected_key],
        "model_path": model_path,
        "reload_verified": True,
    }


if __name__ == "__main__":
    result = run_ml_experiment()
    print(result["metrics"].to_string(index=False))
    print(f"Modèle retenu : {result['selected_model']}")
    print(f"Rechargement vérifié : {result['reload_verified']}")
