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
        Initialize the predictor.
        df : DataFrame containing the data
        target : name of the target column
        """
        self.df = df
        self.target = target
        self.random_state = random_state
        
        # Separate X and y
        self.X = df.drop(target, axis=1)
        self.y = df[target]

        # Split into train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Initialize models
        self.models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(random_state=random_state),
            "XGBoost": xgb.XGBRegressor(random_state=random_state, objective="reg:squarederror")
        }

        # Hyperparameters for GridSearch
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
        
        # Store results
        self.trained_models = {}
        self.rmse_scores = {}

    # -----------------------------
    # Method to train models
    # -----------------------------
    def train(self, model_names=None):
        """
        Train one or more models.
        model_names : str, list of str, or None
                     - None: train all models (default)
                     - str: train a single model ("LinearRegression", "RandomForest", or "XGBoost")
                     - list: train multiple specific models
        """
        # Determine which models to train
        if model_names is None:
            models_to_train = list(self.models.keys())
        elif isinstance(model_names, str):
            models_to_train = [model_names]
        else:
            models_to_train = model_names

        # Validate model names
        for name in models_to_train:
            if name not in self.models:
                raise ValueError(f"Unknown model: {name}. Choose from {list(self.models.keys())}")

        # Train each selected model
        for model_name in models_to_train:
            if model_name == "LinearRegression":
                # 1. Linear Regression (no GridSearch)
                lr = self.models["LinearRegression"]
                lr.fit(self.X_train, self.y_train)
                self.trained_models["LinearRegression"] = lr
                pred = lr.predict(self.X_test)
                self.rmse_scores["LinearRegression"] = np.sqrt(mean_squared_error(self.y_test, pred))
                print(f"LinearRegression trained. RMSE: {self.rmse_scores['LinearRegression']:.2f}")

            elif model_name == "RandomForest":
                # 2. Random Forest with GridSearch
                rf = self.models["RandomForest"]
                rf_grid = GridSearchCV(
                    rf, self.params["RandomForest"], cv=3,
                    scoring="neg_root_mean_squared_error", n_jobs=-1
                )
                rf_grid.fit(self.X_train, self.y_train)
                self.trained_models["RandomForest"] = rf_grid.best_estimator_
                pred = rf_grid.predict(self.X_test)
                self.rmse_scores["RandomForest"] = np.sqrt(mean_squared_error(self.y_test, pred))
                print(f"RandomForest trained. RMSE: {self.rmse_scores['RandomForest']:.2f}")

            elif model_name == "XGBoost":
                # 3. XGBoost with GridSearch
                xgbr = self.models["XGBoost"]
                xgb_grid = GridSearchCV(
                    xgbr, self.params["XGBoost"], cv=3,
                    scoring="neg_root_mean_squared_error", n_jobs=-1
                )
                xgb_grid.fit(self.X_train, self.y_train)
                self.trained_models["XGBoost"] = xgb_grid.best_estimator_
                pred = xgb_grid.predict(self.X_test)
                self.rmse_scores["XGBoost"] = np.sqrt(mean_squared_error(self.y_test, pred))
                print(f"XGBoost trained. RMSE: {self.rmse_scores['XGBoost']:.2f}")

        print("\nTraining complete.")

    # -----------------------------
    # Method to get RMSE scores
    # -----------------------------
    def get_scores(self):
        return pd.DataFrame({
            "Model": list(self.rmse_scores.keys()),
            "RMSE": list(self.rmse_scores.values())
        }).sort_values("RMSE")

    # -----------------------------
    # Method to predict with a given model
    # -----------------------------
    def predict(self, model_name, X_new):
        """
        model_name : "LinearRegression", "RandomForest", "XGBoost"
        X_new : DataFrame or array of new data
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained. Call train() first.")
        model = self.trained_models[model_name]
        return model.predict(X_new)

if __name__ == "__main__":
    # 1. Load data
    df = pd.read_csv("clean_dataset.csv")  # or your existing DataFrame

    # 2. Create the object
    predictor = PricePredictor(df)

    # 3. Train models
    # Option A: Train all models (default)
    predictor.train()

    # Option B: Train a single model
    # predictor.train("RandomForest")

    # Option C: Train specific models
    # predictor.train(["LinearRegression", "RandomForest"])

    # 4. View RMSE scores
    print("\nRMSE Scores:")
    print(predictor.get_scores())

    # 5. Make a prediction
    X_new = predictor.X.iloc[:5]  # First 5 rows (without target column)
    preds = predictor.predict("RandomForest", X_new)
    print("\nPredictions:")
    print(preds)