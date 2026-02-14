# example d'utilisation
import pandas as pd
from model import GaussianNB
import numpy as np
from utils.split import train_test_split

if __name__ == "__main__":
    # Chargement des données
    data = pd.read_csv("./data/flowers_dataset.csv", index_col=False )
    data = data.drop(columns=['id'], errors="ignore")

    # encodange des couleur dominante (one-hot encoding)
    data = pd.get_dummies(data, prefix="couleur", columns=["couleur_dominante"], dtype=int)

    # decoupage en donne de test et entrainement
    X = data.drop(columns=["espece"], axis=0)
    y = data["espece"]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # initialistion et entrainment du model
    model = GaussianNB()
    model.fit(X_train, y_train) # entrainement du model
    # model.load_model("models.json") # chargement du model


    # predictions et evaluation
    y_pred = model.predict(np.array(X_test))

    accuray_score = model.accuracy(y_test, y_pred)
    print(f"accuracy: {accuray_score*100:.2f}%")

    # sauvegarde du model
    # model.save_model("models.json")