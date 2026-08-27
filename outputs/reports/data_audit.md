# Synthèse de l'audit des données

Audit généré sur **5572 messages** et **4 colonnes**.

## Distribution des labels

| Label | Nombre | Pourcentage |
|---|---:|---:|
| ham | 4825 | 86.59 % |
| spam | 747 | 13.41 % |

Ratio majorité/minorité : **6.459:1**.

## Valeurs manquantes

| Colonne | Nombre |
|---|---:|
| id | 0 |
| label | 0 |
| text_fr | 0 |
| text_en | 0 |

## Risques de fuite et doublons

- Doublons exacts hors identifiant : 415
- Groupes dupliqués en français : 295 (438 lignes excédentaires)
- Groupes dupliqués en anglais : 289 (415 lignes excédentaires)
- Textes français associés à des labels différents : 0
- Textes anglais associés à des labels différents : 0

Ces doublons devront être pris en compte lors du futur split afin d'éviter qu'un même message se retrouve dans plusieurs partitions. Le split devra être effectué avant toute opération apprenant à partir des données.

## Statistiques de longueur

Les statistiques complètes (moyenne, médiane, extrêmes et quartiles par classe) sont disponibles dans `data_audit.json` et dans le notebook exécuté.

## Qualité et limites de la traduction française

- Les textes français sont des traductions automatiques du corpus anglais, pas des SMS natifs collectés en France.
- Les formulations, monnaies, numéros et références culturelles du corpus source peuvent subsister.
- L'échantillon bilingue permet une inspection qualitative, pas une mesure de qualité linguistique.
- Aucun split train/test n'est réalisé : il devra précéder toute opération apprenant à partir des données.
- Les licences annoncées par Hugging Face (GPL) et UCI (CC BY 4.0) sont documentées séparément.

Les marqueurs automatiques servent uniquement à orienter l'inspection humaine de l'échantillon reproductible de 10 ham et 10 spam. Ils ne constituent pas une note de qualité linguistique.

## Métriques futures

L'accuracy seule sera insuffisante face au déséquilibre. Les travaux futurs suivront la precision spam, le recall spam, le F1 spam, le F1 macro et la matrice de confusion. Le choix de la métrique prioritaire reste ouvert et sera traité dans l'ADR dédiée.
