# Sources des données

## Dataset utilisé

**SMS Spam Multilingual Collection Dataset**, publié par `dbarbedillo` sur Hugging Face :

- page : https://huggingface.co/datasets/dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset
- fichier récupéré : `data-augmented.csv`, conservé localement sous `data/raw/sms_multilingual.csv` ;
- split annoncé : `train`, 5 572 lignes ;
- colonnes utilisées : `labels`, `text`, `text_fr` ;
- licence annoncée sur la fiche Hugging Face : **GPL** (version non précisée par la fiche).

La fiche du dataset indique que le texte anglais vient du SMS Spam Collection et que les versions française, allemande et hindi proviennent d'un jeu multilingue intermédiaire référencé sur Kaggle. Le corpus multilingue doit donc être traité comme un artefact dérivé, avec sa propre indication de licence, sans remplacer silencieusement la licence du corpus original.

## Dataset original

**SMS Spam Collection**, UCI Machine Learning Repository :

- page : https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- auteurs cités par UCI : Tiago Almeida et José María Gómez Hidalgo ;
- DOI : `10.24432/C5CC84` ;
- licence indiquée par UCI : **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

Les mentions **GPL** du dérivé Hugging Face et **CC BY 4.0** de la source UCI sont consignées séparément. Ce projet ne prétend pas résoudre leur articulation juridique.

## Limite fondamentale

Les textes de `text_fr` sont des **traductions automatiques** des SMS anglais. Ils ne constituent pas un corpus natif représentatif des spams réellement envoyés en France.

La traduction peut conserver ou introduire des formulations littérales, des monnaies britanniques, des numéros et des références culturelles anglophones. Cette limite doit rester visible dans le rapport d'audit, la soutenance et les pistes d'amélioration. Une suite pertinente consistera à collecter et documenter un corpus natif francophone conforme aux contraintes juridiques et de confidentialité.

## Reproductibilité

Date de récupération : 24 août 2026. URL directe utilisée :

`https://huggingface.co/datasets/dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset/resolve/main/data-augmented.csv`

Les CSV contenant les messages ne sont pas redistribués dans le dépôt public. Cette décision évite de présenter comme résolue l'articulation entre les licences annoncées pour le dérivé multilingue et le corpus original. Le téléchargement depuis la source et la préparation locale sont documentés dans `data/README.md`; seul le fichier d'identifiants du split expérimental est versionné.
