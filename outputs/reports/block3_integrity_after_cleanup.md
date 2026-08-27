# Intégrité du Bloc 3 après nettoyage

- Date de contrôle : 25 août 2026
- Dataset dédupliqué : **5 134 SMS**
- Train : **4 107 SMS**
- Test : **1 027 SMS**
- SHA-256 de `ml_split_ids.csv` : `6d1557cb761f1b271f362767ece08dedc55b2f56d28f496bded8dad4b0c97f87`
- Modèles comparés présents dans les métriques : Logistic Regression, Linear SVM, Random Forest
- Modèle sauvegardé rechargeable : **LinearSVC + TF-IDF**
- F1 spam Linear SVM enregistré : **0,944882**
- Prédictions de contrôle : `ham` et `spam`
- Tests après nettoyage : **18 réussis**

## Conclusion

Le nettoyage a retiré uniquement l'ancienne expérimentation Deep Learning et ses dépendances. Le dataset, le split, les métriques, le pipeline Linear SVM et la démonstration Streamlit ML du Bloc 3 restent fonctionnels.

Les artefacts générés retirés ont été déplacés dans une archive locale récupérable, conservée hors du projet et hors du dépôt public.
