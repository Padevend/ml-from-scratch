import numpy as np
import json as js
import pandas as pd

class GaussianNB:
    def __init__(self):
        self.model = {}
    
    def _gaussian_probability(self, x, mean, var):
        return -0.5 * np.log(2 * np.pi * var) - ((x - mean)**2 / (2 * var))

    def _class_score(self, x, prior, mean, var):
        log_prior = np.log(prior)
        log_prob = np.sum(self._gaussian_probability(x, mean, var))
        return log_prob + log_prior

    def fit(self, features, target):
        classes = np.unique(target)

        for c in classes:
            X_c = features[target == c]

            # probabilite des classes P(C)
            prior = len(X_c) / len(features)

            # probabilite par features
            mean = X_c.mean(axis=0)
            var = X_c.var(axis=0) + 1e-9

            self.model[c] = {"prior": prior, "mean": mean, "var": var}

    def predict(self, X):
        prediction = []
        
        for x in X:
            scores = {}

            for c in self.model:
                prior = self.model[c]["prior"]
                mean = self.model[c]["mean"]
                var = self.model[c]["var"]

                score = self._class_score(x, prior, mean, var)
                scores[c] = score
            
            prediction.append(max(scores, key=scores.get))
        
        return np.array(prediction)
    
    def accuracy(self, feature_true, feature_pred):
        return np.sum(feature_true == feature_pred) / len(feature_true)
    
    def load_model(self, filepath: str):
        """Charge les poids du model depuis un fichier JSON"""
        final_df = pd.read_json(filepath, orient="split")
        model = {}
        for c in final_df.index.get_level_values(0).unique():
            df_c = final_df.loc[c]
            model[c] = {
                "mean": df_c["mean"].to_numpy(),
                "var": df_c["var"].to_numpy(),
                "prior": df_c["prior"].iloc[0]
            }
        
        self.model = model
        return model
    
    def save_model(self, filepath: str):
        """Export les paramètres du modèle dans un fichier JSON"""
        df_dict = {}
        for c in self.model:
            n_features = len(self.model[c]["mean"])
            df = pd.DataFrame({
                "mean": self.model[c]["mean"],
                "var": self.model[c]["var"],
                "prior": self.model[c]["prior"] * n_features
            })
            df_dict[c] = df
        final_df = pd.concat(df_dict, names=["class", "feature"])
        final_df.to_json(filepath, indent=4, orient="split")

