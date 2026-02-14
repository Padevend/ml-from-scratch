from model import KNearestClassifier
import numpy as np
import pandas as pd
from utils.split import train_test_split

if __name__ == "__main__":
    # chargement du dataset
    data = pd.read_csv('./data/flowers_dataset.csv', index_col=0)
    data = data.drop(columns=['id'], errors="ignore") # suppression de la colonne id

    # séparation des caractéristiques et des étiquettes
    X = data.drop('espece', axis=1)
    Y = data['espece']

    # normalisation des caractéristiques et categorisation des étiquettes
    enc_col = pd.get_dummies(X, columns=["couleur_dominante"], prefix="couleur", dtype=int)
    X.drop(columns=["couleur_dominante"], inplace=True)
    enc_col = enc_col.drop(X.columns, axis=1)

    # normalisation des donnees numeriques
    mean = X.mean()
    std = X.std()
    X = (X - mean) / std
    X = pd.concat([X, enc_col], axis=1)

    # division des données en ensembles d'entraînement et de test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=47)

    # création et entraînement du modele
    model = KNearestClassifier(k=5)
    model.fit(X_train, Y_train)

    # évaluation du modèle
    accuracy = model.accuracy(X_test, Y_test)
    print(f"Accuracy : {accuracy * 100:.2f}%")

    # nouvelle predisction
    # pred = model.predict(np.array(X_test))
    # print(pred)
