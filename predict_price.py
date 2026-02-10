import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error

class PricePredictor:
    def __init__(self, df, target="buy_price", test_size=0.2, random_state=42):
        """
        Initialise le prédicteur.
        df : DataFrame contenant les données
        target : nom de la colonne cible
        """
        self.df = df
        self.target = target
        self.random_state = random_state
        
        # Séparer X et y
        self.X = df.drop(target, axis=1)
        self.y = df[target]
        
        # Diviser en train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Initialiser les modèles
        self.models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(random_state=random_state),
            "XGBoost": xgb.XGBRegressor(random_state=random_state, objective="reg:squarederror")
        }
        
        # Hyperparamètres pour GridSearch
        self.params = {
            "RandomForest": {
                "n_estimators": [100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5]
            },
            "XGBoost": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5]
            }
        }
        
        # Stocker les résultats
        self.trained_models = {}
        self.rmse_scores = {}

    # -----------------------------
    # Méthode pour entraîner tous les modèles
    # -----------------------------
    def train(self):
        # 1️⃣ Linear Regression (pas de GridSearch)
        lr = self.models["LinearRegression"]
        lr.fit(self.X_train, self.y_train)
        self.trained_models["LinearRegression"] = lr
        pred = lr.predict(self.X_test)
        self.rmse_scores["LinearRegression"] = np.sqrt(mean_squared_error(self.y_test, pred))
        
        # 2️⃣ Random Forest avec GridSearch
        rf = self.models["RandomForest"]
        rf_grid = GridSearchCV(
            rf, self.params["RandomForest"], cv=3, 
            scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        rf_grid.fit(self.X_train, self.y_train)
        self.trained_models["RandomForest"] = rf_grid.best_estimator_
        pred = rf_grid.predict(self.X_test)
        self.rmse_scores["RandomForest"] = np.sqrt(mean_squared_error(self.y_test, pred))
        
        # 3️⃣ XGBoost avec GridSearch
        xgbr = self.models["XGBoost"]
        xgb_grid = GridSearchCV(
            xgbr, self.params["XGBoost"], cv=3, 
            scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        xgb_grid.fit(self.X_train, self.y_train)
        self.trained_models["XGBoost"] = xgb_grid.best_estimator_
        pred = xgb_grid.predict(self.X_test)
        self.rmse_scores["XGBoost"] = np.sqrt(mean_squared_error(self.y_test, pred))
        
        print("✅ Entraînement terminé.")

    # -----------------------------
    # Méthode pour obtenir les scores RMSE
    # -----------------------------
    def get_scores(self):
        return pd.DataFrame({
            "Model": list(self.rmse_scores.keys()),
            "RMSE": list(self.rmse_scores.values())
        }).sort_values("RMSE")

    # -----------------------------
    # Méthode pour prédire avec un modèle donné
    # -----------------------------
    def predict(self, model_name, X_new):
        """
        model_name : "LinearRegression", "RandomForest", "XGBoost"
        X_new : DataFrame ou array des nouvelles données
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modèle {model_name} non entraîné. Appelez d'abord train()")
        model = self.trained_models[model_name]
        return model.predict(X_new)

if __name__ == "__main__":
    df = pd.read_csv("clean_dataset.csv")  # ou ton DataFrame existant

    # 2️⃣ Créer l'objet
    predictor = PricePredictor(df)

    # 3️⃣ Entraîner tous les modèles
    predictor.train()

    # 4️⃣ Voir les RMSE
    print(predictor.get_scores())

    # 5️⃣ Faire une prédiction
    X_new = df.iloc[:5, :-1]  # 5 premières lignes sans la colonne cible
    preds = predictor.predict("RandomForest", X_new)
    print(preds)