# README

## Projet Naive Bayes pour la Classification des Fleurs

Ce projet utilise un classificateur Naive Bayes pour prédire l'espèce de fleurs à partir d'un ensemble de données contenant des caractéristiques des fleurs.

### Installation

Assurez-vous d'avoir Python et les bibliothèques nécessaires installées. Vous pouvez installer les dépendances requises avec :

```bash
pip install pandas numpy
```

### Utilisation

1. **Chargement des Données**: Le script `main.py` charge les données à partir du fichier CSV.
2. **Prétraitement**: Les couleurs dominantes sont encodées en utilisant le one-hot encoding.
3. **Division des Données**: Les données sont divisées en ensembles d'entraînement et de test.
4. **Entraînement du Modèle**: Le modèle Gaussian Naive Bayes est initialisé et entraîné sur les données d'entraînement.
5. **Prédiction et Évaluation**: Le modèle effectue des prédictions sur l'ensemble de test et calcule la précision.
6. **Sauvegarde du Modèle**: Les paramètres du modèle peuvent être sauvegardés dans un fichier JSON.

### Structure du Projet

- `main.py` : Script principal pour l'exécution du pipeline complet
- `model.py` : Classe `GaussianNB` implémentant l'algorithme Naive Bayes
- `utils/split.py` : Fonction pour diviser les données en ensembles d'entraînement et de test
- `data/flowers_dataset.csv` : Ensemble de données contenant les caractéristiques des fleurs

### Modèle Gaussian Naive Bayes

La classe `GaussianNB` implémente un classificateur Naive Bayes basé sur une distribution gaussienne. Elle comprend les méthodes suivantes :

- `fit(features, target)` : Entraîne le modèle en calculant les moyennes, variances et probabilités a priori pour chaque classe
- `predict(X)` : Effectue des prédictions sur de nouvelles données
- `accuracy(feature_true, feature_pred)` : Évalue la précision du modèle
- `save_model(filepath)` : Exporte les paramètres du modèle en JSON
- `load_model(filepath)` : Charge les paramètres du modèle depuis un fichier JSON

### Exemple d'Exécution

Exécutez le script principal :

```bash
python main.py
```

La sortie affichera la précision du modèle en pourcentage.