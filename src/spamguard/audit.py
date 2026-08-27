"""Audit descriptif reproductible du dataset, sans entraînement de modèle."""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from spamguard.config import PROJECT_ROOT, load_config
from spamguard.data import load_processed_data
from spamguard.features import add_exploratory_features
from spamguard.persistence import save_json, save_markdown


def _native(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def _length_statistics(data: pd.DataFrame) -> dict:
    statistics = {}
    for label, group in data.groupby("label", sort=True):
        statistics[label] = {}
        for column in ("n_chars", "n_words"):
            description = group[column].describe(percentiles=[0.25, 0.5, 0.75])
            statistics[label][column] = {
                key: _native(description[key])
                for key in ("count", "mean", "min", "25%", "50%", "75%", "max")
            }
    return statistics


def build_audit_report(data: pd.DataFrame) -> dict:
    enriched = add_exploratory_features(data)
    counts = data["label"].value_counts().sort_index()
    percentages = (counts / len(data) * 100).round(2)
    min_count = int(counts.min())

    fr_groups = data.groupby("text_fr", dropna=False).size()
    en_groups = data.groupby("text_en", dropna=False).size()
    fr_conflicts = data.groupby("text_fr", dropna=False)["label"].nunique()
    en_conflicts = data.groupby("text_en", dropna=False)["label"].nunique()
    exact_subset = ["label", "text_fr", "text_en"]

    cultural_pattern = r"(?i)(?:£|\bpounds?\b|\buk\b|london|wimbledon|nokia|orange)"
    translation_signals = {
        "messages_with_british_currency_marker": int(
            data["text_fr"].str.contains(r"£|\blivres?\b", case=False, regex=True, na=False).sum()
        ),
        "messages_with_anglophone_reference_marker": int(
            data["text_fr"].str.contains(cultural_pattern, regex=True, na=False).sum()
        ),
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "number_of_messages": int(len(data)),
        "number_of_columns": int(data.shape[1]),
        "columns": data.columns.tolist(),
        "unique_labels": sorted(data["label"].dropna().unique().tolist()),
        "label_distribution": {
            label: {"count": int(counts[label]), "percentage": float(percentages[label])}
            for label in counts.index
        },
        "imbalance_ratio_majority_to_minority": round(int(counts.max()) / min_count, 3),
        "missing_values": {column: int(value) for column, value in data.isna().sum().items()},
        "duplicates": {
            "exact_duplicate_rows_excluding_id": int(data.duplicated(exact_subset).sum()),
            "french_duplicate_groups": int((fr_groups > 1).sum()),
            "french_duplicate_excess_rows": int((fr_groups - 1).clip(lower=0).sum()),
            "english_duplicate_groups": int((en_groups > 1).sum()),
            "english_duplicate_excess_rows": int((en_groups - 1).clip(lower=0).sum()),
            "french_texts_with_conflicting_labels": int((fr_conflicts > 1).sum()),
            "english_texts_with_conflicting_labels": int((en_conflicts > 1).sum()),
        },
        "length_statistics": _length_statistics(enriched),
        "exploratory_indicators": {
            label: {
                "url_percentage": round(float(group["has_url"].mean() * 100), 2),
                "number_percentage": round(float(group["has_number"].mean() * 100), 2),
                "mean_special_characters": round(float(group["n_special_chars"].mean()), 2),
                "mean_uppercase_ratio": round(float(group["uppercase_ratio"].mean()), 4),
            }
            for label, group in enriched.groupby("label", sort=True)
        },
        "problems_detected": {
            **translation_signals,
            "note": (
                "Ces comptages sont des signaux observables, pas une mesure automatique "
                "de la qualité linguistique. L'échantillon bilingue doit être examiné humainement."
            ),
        },
        "limits": [
            "Les textes français sont des traductions automatiques du corpus anglais, pas des SMS natifs collectés en France.",
            "Les formulations, monnaies, numéros et références culturelles du corpus source peuvent subsister.",
            "L'échantillon bilingue permet une inspection qualitative, pas une mesure de qualité linguistique.",
            "Aucun split train/test n'est réalisé : il devra précéder toute opération apprenant à partir des données.",
            "Les licences annoncées par Hugging Face (GPL) et UCI (CC BY 4.0) sont documentées séparément.",
        ],
    }


def save_figures(data: pd.DataFrame, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    enriched = add_exploratory_features(data)

    counts = data["label"].value_counts().reindex(["ham", "spam"])
    ax = counts.plot.bar(color=["#3b82f6", "#ef4444"], rot=0, title="Distribution des classes")
    ax.set(xlabel="Label", ylabel="Nombre de messages")
    ax.figure.tight_layout()
    ax.figure.savefig(directory / "class_distribution.png", dpi=150)
    plt.close(ax.figure)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, color in (("ham", "#3b82f6"), ("spam", "#ef4444")):
        values = enriched.loc[enriched["label"] == label, "n_chars"]
        axes[0].hist(values, bins=40, alpha=0.55, label=label, color=color)
    axes[0].set(title="Longueur en caractères", xlabel="Caractères", ylabel="Fréquence")
    axes[0].legend()
    enriched.boxplot(column="n_words", by="label", ax=axes[1], grid=False)
    axes[1].set(title="Longueur en mots par classe", xlabel="Label", ylabel="Mots")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(directory / "message_lengths.png", dpi=150)
    plt.close(fig)

    indicators = enriched.groupby("label")[["has_url", "has_number"]].mean().mul(100)
    ax = indicators.plot.bar(rot=0, color=["#8b5cf6", "#f59e0b"], title="Indicateurs exploratoires")
    ax.set(xlabel="Label", ylabel="Messages concernés (%)")
    ax.figure.tight_layout()
    ax.figure.savefig(directory / "exploratory_indicators.png", dpi=150)
    plt.close(ax.figure)


def report_to_markdown(report: dict) -> str:
    distribution = report["label_distribution"]
    duplicates = report["duplicates"]
    missing = report["missing_values"]
    lines = [
        "# Synthèse de l'audit des données",
        "",
        f"Audit généré sur **{report['number_of_messages']} messages** et **{report['number_of_columns']} colonnes**.",
        "",
        "## Distribution des labels",
        "",
        "| Label | Nombre | Pourcentage |",
        "|---|---:|---:|",
    ]
    for label, values in distribution.items():
        lines.append(f"| {label} | {values['count']} | {values['percentage']:.2f} % |")
    lines += [
        "",
        f"Ratio majorité/minorité : **{report['imbalance_ratio_majority_to_minority']:.3f}:1**.",
        "",
        "## Valeurs manquantes",
        "",
        "| Colonne | Nombre |",
        "|---|---:|",
        *[f"| {column} | {count} |" for column, count in missing.items()],
        "",
        "## Risques de fuite et doublons",
        "",
        f"- Doublons exacts hors identifiant : {duplicates['exact_duplicate_rows_excluding_id']}",
        f"- Groupes dupliqués en français : {duplicates['french_duplicate_groups']} ({duplicates['french_duplicate_excess_rows']} lignes excédentaires)",
        f"- Groupes dupliqués en anglais : {duplicates['english_duplicate_groups']} ({duplicates['english_duplicate_excess_rows']} lignes excédentaires)",
        f"- Textes français associés à des labels différents : {duplicates['french_texts_with_conflicting_labels']}",
        f"- Textes anglais associés à des labels différents : {duplicates['english_texts_with_conflicting_labels']}",
        "",
        "Ces doublons devront être pris en compte lors du futur split afin d'éviter qu'un même message se retrouve dans plusieurs partitions. Le split devra être effectué avant toute opération apprenant à partir des données.",
        "",
        "## Statistiques de longueur",
        "",
        "Les statistiques complètes (moyenne, médiane, extrêmes et quartiles par classe) sont disponibles dans `data_audit.json` et dans le notebook exécuté.",
        "",
        "## Qualité et limites de la traduction française",
        "",
        *[f"- {limit}" for limit in report["limits"]],
        "",
        "Les marqueurs automatiques servent uniquement à orienter l'inspection humaine de l'échantillon reproductible de 10 ham et 10 spam. Ils ne constituent pas une note de qualité linguistique.",
        "",
        "## Métriques futures",
        "",
        "L'accuracy seule sera insuffisante face au déséquilibre. Les travaux futurs suivront la precision spam, le recall spam, le F1 spam, le F1 macro et la matrice de confusion. Le choix de la métrique prioritaire reste ouvert et sera traité dans l'ADR dédiée.",
    ]
    return "\n".join(lines) + "\n"


def run_audit() -> dict:
    config = load_config()
    data = load_processed_data()
    report = build_audit_report(data)
    report_dir = PROJECT_ROOT / config["paths"]["reports"]
    save_json(report, report_dir / "data_audit.json")
    save_markdown(report_to_markdown(report), report_dir / "data_audit.md")
    save_figures(data, PROJECT_ROOT / config["paths"]["figures"])
    return report


if __name__ == "__main__":
    result = run_audit()
    print(f"Audit généré pour {result['number_of_messages']} messages.")
