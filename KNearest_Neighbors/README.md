
# K-Nearest Neighbors (KNN)

## Aperçu
K-Nearest Neighbors est un algorithme d'apprentissage automatique supervisé simple utilisé pour les tâches de classification et de régression. Il classifie les points de données en fonction des classes de leurs voisins les plus proches.

## Concepts Clés
- **Valeur K** : Nombre de voisins les plus proches à considérer
- **Métrique de Distance** : Distance euclidienne
- **Entraînement** : Stocke toutes les données d'entraînement (apprentissage paresseux)
- **Prédiction** : Trouve les K points les plus proches et utilise le vote majoritaire (classification) ou la valeur moyenne (régression)

## Quand l'Utiliser
- Petits à moyens ensembles de données
- Problèmes de classification et régression non-linéaires
- Comparaison de modèles de base

## Implémentation

### Exemple de Classification
```python
from model import KNearestClassifier
from utils.split import train_test_split

model = KNearestClassifier(k=5)
model.fit(X_train, Y_train)
accuracy = model.accuracy(X_test, Y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

### Exemple de Régression
```python
from model import KNearestRegressor

model = KNearestRegressor(k=5)
model.fit(X_train, Y_train)
score = model.score(X_test, Y_test)
predictions = model.predict(new_data)
```

## Paramètres

- `k` : nombre de voisin a considerer

## Exemple

Voir `example_classifier.py` pour un exemple d'implémentation de classification et `example_regressor.py` pour un exemple d'implementation de regressement

## Plus d'information

[Machine learning: k-Nearest Neighbors](https://www.mbah-ndam.dev/ml/ml-k-nearest-neighbors)

