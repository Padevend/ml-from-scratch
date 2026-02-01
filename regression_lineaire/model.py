import pandas as pd
import numpy as np


class LinearRegresaaion:
    def __init__(self, rate=0.001, precision=10e-6, max_iter=10000):
        """Initialisation du model de regression lineaire"""
        # hyperparametres
        self.rate = rate
        self.precision = precision
        self.max_iter = max_iter
    
    def _normalize(self, features):
        """Normalisation des features par standardisation"""

        # éviter la division par zéro
        std = np.where(self.std == 0, 1, self.std)
        normalized_features = (features - self.mean) / std
        return normalized_features, self.mean, std

    def fit(self, features: list = None, target: list = None):
        """
        Entraînement du modèle par descente de gradient.
        """
        # Vérification des entrées
        if features is None or target is None:
            raise ValueError("Features et target doivent être fournis")
        
        # Conversion en numpy array
        self.features = np.array(features)
        self.target = np.array(target)

         # normalisation des dimensions
        self.mean = self.features.mean(axis=0)
        self.std = self.features.std(axis=0)
        self.features, _, _ = self._normalize(self.features)

        m, n = self.features.shape

        # Initialisation des poids (n features + biais)
        self.weight = np.random.random(n + 1)

        # Création de la matrice avec biais
        self.feat_b = np.c_[self.features, np.ones(m)]  # shape (m, n+1)

        # Descente de gradient
        for i in range(self.max_iter):
            # Prédictions
            y_pred = self.feat_b @ self.weight

            # Gradient
            grad = (2 / m) * self.feat_b.T @ (y_pred - self.target)

            # Convergence basée sur la norme du gradient
            if np.linalg.norm(grad) < self.precision:
                print(f"Converged after {i} iterations")
                break

            # Mise à jour des poids
            self.weight -= self.rate * grad
        else:
            print(f"Max iterations reached ({self.max_iter})")

    def score(self, features=None, target=None):
        """Retourne l'erreur quadratique moyenne du model"""
        if not hasattr(self, "weight"):
            raise ValueError("Le model n'est pas encore entrainé")

        # mean square error
        if features is not None and target is not None:
            #normalisation des dimensions
            mean = features.mean(axis=0)
            std = features.std(axis=0)
            # éviter la division par zéro
            std = np.where(std == 0, 1, std)
            features = (features - mean) / std
            
            feat_b = np.c_[features, np.ones(len(features))]
            y_pred = feat_b @ self.weight
            error = np.mean((target - y_pred)**2)
        else:
            y_pred = self.feat_b @ self.weight
            error = np.mean((self.target - y_pred)**2)
        
        return np.sqrt(error)

    def predict(self, features: list):
        """
        Prédiction du modèle pour de nouvelles données.
        """
        if not hasattr(self, "weight") or self.weight is None:
            raise ValueError("Le modèle n'est pas encore entraîné")

        # Conversion en numpy array
        predict_features = np.array(features, dtype=float)

        # Normalisation avec mean et std calculés lors du fit
        if not hasattr(self, "mean") or not hasattr(self, "std"):
            # Calcul et stockage lors du fit
            self.mean = self.features.mean(axis=0)
            self.std = self.features.std(axis=0)
            self.std = np.where(self.std == 0, 1, self.std)

        predict_features = (predict_features - self.mean) / self.std

        # Vérification des dimensions
        predict_m, predict_n = predict_features.shape

        if self.weight.size != predict_n + 1:  # +1 pour le biais
            raise ValueError(
                f"Le nombre de features ({predict_n}) ne correspond pas au modèle ({self.weight.size - 1})"
            )

        # Ajout du biais et calcul de la prédiction
        predict = np.c_[predict_features, np.ones(predict_m)] @ self.weight

        return predict

    def get_weights(self):
        """Retourne les poids du model"""
        if not hasattr(self, "weight"):
            raise ValueError("Le model n'est pas encore entrainé")

        return self.weight

    def load_model(self, filepath: str):
        """Charge les poids du model depuis un fichier JSON"""
        data = pd.read_json(filepath, typ="series")

        self.weight = np.array(data["weight"])
        self.mean = np.array(data["mean"]) if "mean" in data else None
        self.std = np.array(data["std"]) if "std" in data else None 

        return self.weight

    def save_model(self, filepath: str):
        """Export les paramètres du modèle dans un fichier JSON"""
        weights = pd.Series({
            "weight": self.weight.tolist(),
            "mean": self.mean.tolist() if hasattr(self, "mean") else None,
            "std": self.std.tolist() if hasattr(self, "std") else None
        })

        weights.to_json(filepath, index=False, indent=4)
