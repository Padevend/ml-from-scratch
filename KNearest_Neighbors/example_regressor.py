from model import KNearestRegressor
import numpy as np
import pandas as pd
from utils.split import train_test_split

if __name__ == "__main__":
    # chargement du dataset
    data = pd.read_csv('./data/houses_dataset.csv')

    # séparation des caractéristiques et des étiquettes
    X = data.drop('prix', axis=1)
    Y = data['prix']

    # normalisation des donnees numeriques
    mean = X.mean().to_numpy()
    std = X.std().to_numpy()
    X = (X - mean) / std

    # division des données en ensembles d'entraînement et de test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=47)

    # création et entraînement du modele
    model = KNearestRegressor(k=5)
    model.fit(X_train, Y_train)

    # évaluation du modèle
    score = model.score(X_test, Y_test)
    print(f"Score : {score:.2f}")

    # ---------------------------------------- prédictions sur de nouvelles données ----------------------------------------
    predict_data = np.array([[94, 4, 28], [148, 2, 8]])

    # normalisation des nouvelles données
    predict_data = (predict_data - mean) / std


    predictions = model.predict(predict_data)
    for i, price in enumerate(predictions):
        print(f"Prix prédit pour l'échantillon {i+1} : {price:.2f}")
