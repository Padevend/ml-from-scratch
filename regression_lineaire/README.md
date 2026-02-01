
# Modèle de Régression Linéaire

Une implémentation Python de la régression linéaire avec optimisation par descente de gradient.

## Caractéristiques

- Entraînement par descente de gradient avec détection de convergence
- Normalisation des caractéristiques (standardisation)
- Persistance du modèle (sauvegarde/chargement JSON)
- Évaluation MSE et prédictions

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Entraîner un Modèle

```python
from model import LinearRegresaaion
import pandas as pd

# Charger les données
data = pd.read_csv("./data/houses_dataset.csv")
X = data[["surface", "chambres", "age"]]
y = data["prix"]

# Créer et entraîner le modèle
model = LinearRegresaaion(rate=0.01, max_iter=10000)
model.fit(X, y)
```

### Faire des Prédictions

```python
predictions = model.predict([[94, 4, 28], [148, 2, 8]]) 
print(f"Prédictions : {predictions}")
```

### Évaluation du Modèle

```python
mse = model.score(X, y)
print(f"Erreur Quadratique Moyenne : {mse}")
```

### Sauvegarder/Charger le Modèle

```python
# Sauvegarder
model.save_model("linear_model.json")

# Charger
model.load_model("linear_model.json")
```

## Paramètres

- `rate` : Taux d'apprentissage (par défaut : 0.001)
- `precision` : Seuil de convergence (par défaut : 10e-6)
- `max_iter` : Nombre maximum d'itérations (par défaut : 10000)

## Exemple

Voir `main.py` pour un exemple d'implémentation complet.

## Plus d'information

[Machine learning: model de regression lineare](https://www.mbah-ndam.dev/ml/ml-linear-regression)