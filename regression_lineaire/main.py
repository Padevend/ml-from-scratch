# Exemple d'utilisation
import pandas as pd
from model import LinearRegresaaion

if __name__ == "__main__":
    # Chargement des données
    data = pd.read_csv("./data/houses_dataset.csv")
    X = data[["surface", "chambres", "age"]]
    y = data["prix"]

    # Création et entraînement du modèle
    model = LinearRegresaaion(rate=0.01, max_iter=10000)
    # model.fit(X, y)
    model.load_model("linear_model.json")

    # Évaluation du modèle
    mse = model.score(X, y)
    print(f"Erreur quadratique moyenne: {mse}")

    # sauvegarde du modèle en json
    model.save_model("linear_model.json")

    # Prédiction avec de nouvelles données
    predictions = model.predict([[94, 4, 28], [148, 2, 8]])
    print(f"Prédictions: {predictions}")
