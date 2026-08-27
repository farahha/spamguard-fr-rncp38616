# ADR-001 — Sélection de la métrique prioritaire

- Statut : accepté
- Date : 2026-08-24

## Contexte

Avant déduplication, le corpus comporte 86,59 % de ham et 13,41 % de spam. Une prédiction systématique de la classe majoritaire obtiendrait donc une accuracy apparemment élevée sans répondre au besoin de détection. Un faux positif masque un message légitime ; un faux négatif laisse parvenir un spam potentiellement frauduleux.

## Résultats observés sur le test

| model               |   accuracy |   precision_spam |   recall_spam |   f1_spam |   f1_macro |   false_positives |   false_negatives |
|:--------------------|-----------:|-----------------:|--------------:|----------:|-----------:|------------------:|------------------:|
| Logistic Regression |     0.9815 |           0.9098 |        0.9453 |    0.9272 |     0.9583 |                12 |                 7 |
| Linear SVM          |     0.9864 |           0.9524 |        0.9375 |    0.9449 |     0.9686 |                 6 |                 8 |
| Random Forest       |     0.9776 |           0.9200 |        0.8984 |    0.9091 |     0.9482 |                10 |                13 |

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
