# ADR-002 — Sélection du modèle Machine Learning classique

- Statut : accepté
- Date : 2026-08-24
- Niveau de confiance : modéré

## Contexte et critères

Trois pipelines TF-IDF ont été évalués sur le même test historique : Logistic Regression, Linear SVM et Random Forest. Les critères sont le F1 spam principal, la precision et le recall spam, les faux positifs/négatifs, la simplicité, les temps d'entraînement et d'inférence, et l'interprétabilité.

## Avantages et inconvénients

- **Logistic Regression** : linéaire, coefficients inspectables et compromis de classes réglable ; peut être moins performante qu'un SVM sur un espace sparse très dimensionnel.
- **Linear SVM** : particulièrement adapté aux vecteurs sparse de grande dimension et souvent efficace en classification de texte ; scores non probabilistes et interprétation moins immédiate qu'une régression logistique.
- **Random Forest** : représente la famille des ensembles d'arbres étudiée dans la formation, capture des interactions non linéaires et fournit des importances ; il est plus coûteux, moins compact et n'est pas naturellement optimal pour un TF-IDF sparse de grande dimension.

## Décision

Le pipeline retenu est **Linear SVM**. Sur cette exécution, il obtient un F1 spam de **0.9449**, une precision spam de **0.9524**, un recall spam de **0.9375**, avec **6 faux positifs** et **8 faux négatifs**.

Il présente la meilleure accuracy, la meilleure precision spam et le meilleur F1 spam des trois modèles. Logistic Regression obtient un recall spam légèrement supérieur, mais Linear SVM offre le meilleur compromis global et convient particulièrement aux représentations textuelles TF-IDF sparse de grande dimension.

## Conséquences

Seul ce pipeline TF-IDF complet est sauvegardé dans `models/ml_spam_classifier.joblib`. Les deux autres restent reproductibles via le code mais ne sont pas persistés. Le niveau de confiance est modéré car la comparaison repose sur un seul split et sur un corpus français traduit automatiquement ; la décision devra être réévaluée avec validation croisée et données françaises natives.
