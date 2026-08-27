# Données

Les fichiers CSV contenant les SMS ne sont pas redistribués dans le dépôt public. La fiche Hugging Face annonce une licence GPL pour le dérivé multilingue et UCI une licence CC BY 4.0 pour le corpus original ; leur articulation n'étant pas tranchée ici, le projet fournit la provenance et la procédure de récupération plutôt qu'une copie des textes.

Depuis la racine du projet :

```bash
mkdir -p data/raw data/processed
curl -L \
  "https://huggingface.co/datasets/dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset/resolve/main/data-augmented.csv" \
  -o data/raw/sms_multilingual.csv
python -m spamguard.data
```

Le notebook d'audit génère les rapports descriptifs. Le notebook ML recharge le dataset dédupliqué et le fichier public `processed/ml_split_ids.csv`, sans recréer le split.

- `raw/sms_multilingual.csv` est une copie non modifiée de `data-augmented.csv`, téléchargée depuis le dépôt Hugging Face `dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset`.
- `processed/sms_fr.csv` sélectionne uniquement les colonnes utiles et les renomme en `id`, `label`, `text_fr`, `text_en`.
- `processed/sms_fr_deduplicated.csv` conserve la première occurrence de chaque `text_fr` strictement identique pour les expériences ML.
- `processed/ml_split_ids.csv` fige l'affectation train/test (`80/20`, stratifiée, `random_state=42`) commune aux trois modèles classiques.

Les contenus, labels, accents, ponctuation, chiffres, URL et majuscules ne sont pas modifiés. L'identifiant `sgfr-NNNNNN` correspond à la position de la ligne dans le fichier source et reste stable tant que ce fichier et son ordre ne changent pas.

Important : `text_fr` est une traduction automatique du texte anglais. Il ne s'agit pas d'un corpus de SMS français natifs collectés en France. Voir `docs/data-sources.md` pour la provenance et les licences annoncées.

Le fichier original `sms_fr.csv` reste intact. L'audit comptait 415 doublons exacts sur le triplet texte anglais/texte français/label ; la déduplication expérimentale fondée sur le seul `text_fr` retire 438 lignes, car différents textes anglais peuvent partager une même traduction française.
