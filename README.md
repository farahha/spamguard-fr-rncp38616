# SpamGuard-FR

**Certification RNCP38616 — Bloc 3 · Machine Learning**

## Objectif

Classifier des SMS traduits en français en deux catégories :

- `ham` : message légitime ;
- `spam` : message indésirable.

Ce projet compare trois algorithmes classiques sur une représentation TF-IDF commune et sur un même split train/test figé.

## Dataset

Le projet s'appuie sur le **SMS Spam Multilingual Collection Dataset**, disponible sur Hugging Face et dérivé du corpus original **SMS Spam Collection** publié par UCI (DOI [`10.24432/C5CC84`](https://doi.org/10.24432/C5CC84)). La fiche Hugging Face annonce une licence GPL pour le dérivé, tandis qu'UCI indique CC BY 4.0 pour le corpus original ; ces mentions sont consignées séparément dans la [documentation des sources](docs/data-sources.md).

Le corpus initial contient 5 572 messages : 4 825 ham (86,59 %) et 747 spam (13,41 %). La déduplication stricte de `text_fr` retire 438 lignes et produit 5 134 observations. Le split stratifié figé contient 4 107 messages d'entraînement et 1 027 messages de test.

Les textes français sont issus d'une **traduction automatique** du corpus anglais. Ils ne constituent pas un corpus natif collecté en France et ne représentent pas nécessairement les spams français actuels.

Les CSV contenant les SMS ne sont pas redistribués dans ce dépôt public par prudence sur l'articulation des licences. Le fichier source et la procédure de préparation sont décrits dans [data/README.md](data/README.md). Le fichier d'identifiants [`ml_split_ids.csv`](data/processed/ml_split_ids.csv) est conservé pour figer le split expérimental.

## Pipeline

```text
SMS
→ déduplication
→ split stratifié figé
→ TF-IDF
→ Logistic Regression / Linear SVM / Random Forest
→ comparaison
→ Linear SVM retenu
```

## Résultats

Les métriques ci-dessous proviennent uniquement du test figé de 1 027 messages.

| Modèle | Accuracy | Precision spam | Recall spam | F1 spam |
|---|---:|---:|---:|---:|
| Logistic Regression | 0,9815 | 0,9098 | 0,9453 | 0,9272 |
| **Linear SVM** | **0,9864** | **0,9524** | 0,9375 | **0,9449** |
| Random Forest | 0,9776 | 0,9200 | 0,8984 | 0,9091 |

Ces valeurs décrivent cette expérience sur ce corpus traduit ; elles ne sont pas des performances universelles sur tous les SMS français.

## Modèle retenu

Le modèle final est **Linear SVM** : il obtient le meilleur F1 spam, conserve un bon recall spam, limite les faux positifs à 6 et présente un faible coût d'inférence. Le pipeline TF-IDF + LinearSVC sauvegardé se trouve dans [`models/ml_spam_classifier.joblib`](models/ml_spam_classifier.joblib).

## Parcours Machine Learning

### Notebook 01 — Audit des données

- contrôle de la qualité des données ;
- définition des classes `ham` et `spam` ;
- observation du déséquilibre des classes ;
- détection des doublons ;
- prévention de la fuite de données (*data leakage*).

### Notebook 02 — Baseline

- déduplication ;
- split train/test stratifié (80 % / 20 %, `random_state=42`) ;
- vectorisation TF-IDF ;
- Logistic Regression ;
- entraînement avec `.fit()` et prédiction avec `.predict()` ;
- optimisation de la log-loss ;
- Accuracy, Precision, Recall, F1-score et matrice de confusion.

### Notebook 03 — Comparaison

Comparaison de Logistic Regression, Linear SVM et Random Forest avec le même split et la même représentation TF-IDF. Le modèle retenu est **Linear SVM**.

| Modèle | Accuracy | Precision spam | Recall spam | F1-score spam |
|---|---:|---:|---:|---:|
| Logistic Regression | 98,15 % | 90,98 % | 94,53 % | 92,72 % |
| **Linear SVM** | **98,64 %** | **95,24 %** | 93,75 % | **94,49 %** |
| Random Forest | 97,76 % | 92,00 % | 89,84 % | 90,91 % |

## Correspondance avec les compétences RNCP

- **C1 — Sélectionner l'algorithme le plus adapté** : Notebook 03, comparaison de Logistic Regression, Linear SVM et Random Forest, puis sélection du Linear SVM.
- **C2 — Préparer et transformer les données** : Notebooks 01 et 02, audit, déduplication, split train/test et TF-IDF.
- **C3 — Entraîner un modèle en optimisant une loss function** : Notebook 02, Logistic Regression, `.fit()`, log-loss et ajustement des coefficients.

## Métriques

- **Accuracy** : part totale des messages correctement classés.
- **Precision spam** : part des prédictions spam qui sont réellement des spams.
- **Recall spam** : part des spams réellement détectés.
- **F1 spam** : moyenne harmonique de la precision et du recall spam ; métrique principale de sélection.
- **Matrice de confusion** : répartition des vrais/faux positifs et vrais/faux négatifs.

## Structure du projet

```text
app/                 démonstration Streamlit
config/              configuration du pipeline
data/                 documentation et identifiants du split
docs/adr/             décisions d'architecture
models/               pipeline Linear SVM validé
notebooks/            audit et comparaison ML exécutés
outputs/              figures, métriques et rapports
src/spamguard/         code de préparation, entraînement et inférence
tests/                tests automatisés
```

## Installation

```bash
git clone https://github.com/farahha/spamguard-fr-rncp38616.git
cd spamguard-fr-rncp38616
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Télécharger ensuite le CSV tiers en suivant [data/README.md](data/README.md). Le modèle publié permet toutefois de lancer directement l'inférence et Streamlit sans redistribuer le corpus.

## Exécuter les notebooks

Les trois notebooks du parcours ML sont exécutés et conservent leurs sorties utiles :

- [01 — Audit des données](notebooks/01_data_audit.ipynb)
- [02 — Baseline Logistic Regression](notebooks/02_ml_baseline.ipynb)
- [03 — Comparaison Machine Learning](notebooks/03_ml_comparison.ipynb)

Après préparation des données, lancer Jupyter avec :

```bash
make notebook
```

La commande `make ml` réexécute la comparaison ; elle recharge le split existant au lieu d'en créer un nouveau.

## Tests

```bash
python -m pytest
```

Les tests vérifient notamment l'intégrité du split, l'absence de fuite train/test, le rechargement du pipeline et des prédictions limitées à `ham`/`spam`.

## Démonstration Streamlit

```bash
make app
```

LinearSVC ne produit pas de probabilité calibrée : l'interface présente son score comme une distance à la frontière de décision.

## Artefacts

- [Comparaison CSV](outputs/metrics/ml_model_comparison.csv)
- [Rapport de comparaison](outputs/reports/ml_model_comparison.md)
- [Figure de comparaison](outputs/figures/ml_model_comparison.png)
- [Matrice — Logistic Regression](outputs/figures/confusion_matrix_logistic_regression.png)
- [Matrice — Linear SVM](outputs/figures/confusion_matrix_linear_svm.png)
- [Matrice — Random Forest](outputs/figures/confusion_matrix_random_forest.png)
- [ADR-001 — choix de la métrique](docs/adr/ADR-001-metric-selection.md)
- [ADR-002 — sélection du modèle](docs/adr/ADR-002-ml-model-selection.md)

## Limites

- corpus français traduit automatiquement et non collecté nativement en France ;
- déséquilibre important entre ham et spam ;
- corpus historique de petite taille ;
- performances non garanties sur des SMS contemporains ou hors distribution ;
- LinearSVC non calibré en probabilités.
