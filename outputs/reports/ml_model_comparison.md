# Comparaison des modèles Machine Learning

Les trois pipelines rechargent les identifiants du split historique et ajustent leur TF-IDF uniquement sur le train.

## Données et split

- Avant déduplication : 5572 messages
- Après déduplication stricte sur `text_fr` : 5134 messages
- Lignes supprimées : 438
- Train : 4107 messages ; test : 1027 messages
- Train — ham : 3595, spam : 512
- Test — ham : 899, spam : 128
- Intersection exacte des textes entre train et test : 0

L'audit initial trouvait 415 doublons du triplet complet. La règle expérimentale plus large, fondée uniquement sur `text_fr`, retire davantage de lignes quand plusieurs originaux anglais aboutissent à la même traduction française.

## Résultats sur le test uniquement

| model               |   accuracy |   precision_spam |   recall_spam |   f1_spam |   f1_macro |   false_positives |   false_negatives |   training_time_seconds |   inference_time_ms_mean |
|:--------------------|-----------:|-----------------:|--------------:|----------:|-----------:|------------------:|------------------:|------------------------:|-------------------------:|
| Logistic Regression |   0.981500 |         0.909774 |      0.945312 |  0.927203 |   0.958303 |                12 |                 7 |                0.059992 |                 0.007966 |
| Linear SVM          |   0.986368 |         0.952381 |      0.937500 |  0.944882 |   0.968552 |                 6 |                 8 |                0.057690 |                 0.007525 |
| Random Forest       |   0.977605 |         0.920000 |      0.898438 |  0.909091 |   0.948160 |                10 |                13 |                0.386160 |                 0.032890 |

TP/TN/FP/FN utilisent `spam` comme classe positive. Un faux positif est un SMS légitime écarté ; un faux négatif est un spam qui atteint l'utilisateur.

| Modèle | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic Regression | 887 | 12 | 7 | 121 |
| Linear SVM | 893 | 6 | 8 | 120 |
| Random Forest | 889 | 10 | 13 | 115 |

## Décision

Le modèle retenu est **Linear SVM**, sélectionné sur le F1 spam, avec contrôle conjoint de la precision, du recall, des faux positifs, des faux négatifs et des temps d'exécution.

La métrique principale retenue est le **F1 spam**. Elle équilibre la protection contre les spams non détectés et la préservation des messages légitimes. Le recall spam et le nombre de faux positifs restent des garde-fous obligatoires.

## Exemples d'erreurs du modèle retenu

### Faux positifs
- Pareil pour u...
- Vous n'avez pas d'offre de message
- Avez-vous posé votre ligne d'aérotel pour vous reposer?
- Avez-vous pratiqué votre curtsey?
- Bonjour, Mobile no. &lt;#&gt; vous a ajouté dans leur liste de contacts sur www.fullonsms.com C'est un excellent endroit pour envoyer des sms gratuits aux gens Pour plus de visite fullonsms.com

### Faux négatifs
- Bonjour chéri comment allez-vous aujourd'hui? J'aimerais avoir une conversation, pourquoi ne me dites-vous pas à quoi vous ressemblez et ce que vous êtes dans sexy?
- Salut je suis poursuivi. J'ai 20 ans et je travaille comme un lapdancer. J'aime le sexe. Text me live - Je suis ma chambre maintenant. text SUE à 89555. Par TextOperator G2 1DA 150ppmsg 18+
- Dans The Simpsons Movie sorti en juillet 2007 nom du groupe qui est mort au début du film? A-Green Day, B-Blue Day, C-Red Day. (Envoyer A, B ou C)
- L'alerte d'appel manquée. Ces numéros ont appelé mais n'ont laissé aucun message. 07008009200
- Salut son LUCY Hubby aux meetins toute la journée Fri & I sera B seul à l'hôtel U fantaisie cumin plus? Pls laisser msg 2day 09099726395 Lucy x Calls£1/minMobsmoreLKPOBOX177HP51FL

Ces messages sont présentés pour inspection qualitative seulement. Une explication n'est retenue que si elle est directement visible dans le texte (brièveté, chiffres, URL, ambiguïté ou traduction maladroite).

## Limites

- Les textes français sont des traductions automatiques et non un corpus natif français.
- Le jeu de test n'est utilisé qu'une fois pour cette comparaison, mais une future optimisation exigera une validation distincte ou une validation croisée sur le train.
- Les paramètres TF-IDF et modèles sont des configurations raisonnables, pas des optimums démontrés.
- Les temps dépendent de cette machine et de la charge du système.
