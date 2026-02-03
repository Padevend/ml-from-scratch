import numpy as np

# k-NN en tant que model de classification
class KNearestClassifier:
    def __init__(self, k=3):
        self.k = k
    
    def _predict(self, x):
        # calcul de la distance euclidienne
        dist = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

        # recuperations les indices des k plus proches voisins
        k_indices = np.argsort(dist)[:self.k]

        # recuperation des labels des k plus proches voisins
        k_nearest_labels = self.y_train.iloc[k_indices]
        most_common = k_nearest_labels.mode()[0]

        return most_common
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(np.array(x)) for x in X]
        return np.array(predictions)
    
    def accuracy(self, X_pred, y_pred):
        y_true = self.predict(np.array(X_pred))

        # calcul de l'accuracy
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.sum(y_true == y_pred) / len(y_true)

# k-NN en tant model de regression
class KNearestRegressor:
    def __init__(self, k=3):
        self.k = k
    
    def _predict(self, x):
        # calcul de la distance euclidienne
        dist = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

        # recuperations les indices des k plus proches voisins
        k_indices = np.argsort(dist)[:self.k]

        # recuperation de la moyenne des valeurs des k plus proches voisins
        k_nearest_values = self.y_train.iloc[k_indices]
        mean_value = k_nearest_values.mean()

        return mean_value
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(np.array(x)) for x in X]
        return np.array(predictions)
    
    def score(self, X_pred, y_true):
        y_pred = self.predict(np.array(X_pred))

        # calcul de l'erreur quadratique moyenne
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean((y_true - y_pred) ** 2)