"""
Model Training Module for Madrid House Price Prediction

Core functions to train and save a Random Forest model for predicting house prices.
"""

import pandas as pd
import numpy as np
import pickle
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_and_prepare_data(csv_path: str = 'houses_madrid.csv') -> Tuple[pd.DataFrame, pd.Series, List[str], SimpleImputer]:
    """
    Load data, prepare features, and handle missing values.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file

    Returns:
    --------
    tuple
        (X_imputed, y, feature_columns, imputer)
    """
    # Load data
    df = pd.read_csv(csv_path)
    df = df[df['is_buy_price_known'] == True].copy()

    # Define features
    feature_columns = [
        'sq_mt_built', 'sq_mt_useful', 'n_rooms', 'n_bathrooms', 'n_floors',
        'built_year', 'has_central_heating', 'has_individual_heating',
        'are_pets_allowed', 'has_ac', 'has_fitted_wardrobes', 'has_lift',
        'is_exterior', 'has_garden', 'has_pool', 'has_terrace', 'has_balcony',
        'has_storage_room', 'is_furnished', 'is_kitchen_equipped',
        'is_accessible', 'has_green_zones', 'has_parking',
        'is_parking_included_in_price', 'parking_price',
        'is_orientation_north', 'is_orientation_west',
        'is_orientation_south', 'is_orientation_east',
        'is_renewal_needed', 'is_new_development'
    ]

    # Prepare features and target
    X = df[feature_columns].copy()
    y = df['buy_price'].copy()

    # Remove completely empty columns
    cols_to_drop = X.columns[X.isnull().all()].tolist()
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)
        feature_columns = [col for col in feature_columns if col not in cols_to_drop]

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    return X_imputed, y, feature_columns, imputer


def train_and_evaluate(X: pd.DataFrame,
                       y: pd.Series,
                       test_size: float = 0.2,
                       random_state: int = 42) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """
    Train Random Forest model and evaluate performance.

    Parameters:
    -----------
    X : pd.DataFrame
        Features (already imputed)
    y : pd.Series
        Target variable
    test_size : float
        Proportion of test set
    random_state : int
        Random seed

    Returns:
    --------
    tuple
        (model, metrics)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_test = model.predict(X_test)
    metrics = {
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_r2': r2_score(y_test, y_pred_test)
    }

    return model, metrics


def save_model(model: RandomForestRegressor,
               imputer: SimpleImputer,
               feature_columns: List[str],
               metrics: Dict[str, float],
               output_path: str = 'price_prediction_model.pkl') -> None:
    """
    Save trained model to pickle file.

    Parameters:
    -----------
    model : RandomForestRegressor
        Trained model
    imputer : SimpleImputer
        Fitted imputer
    feature_columns : list
        List of feature names
    metrics : dict
        Model performance metrics
    output_path : str
        Path to save the model
    """
    model_data = {
        'model': model,
        'imputer': imputer,
        'feature_columns': feature_columns,
        'model_type': 'RandomForestRegressor',
        'target': 'buy_price',
        'metrics': metrics
    }

    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)


def train_pipeline(csv_path: str = 'houses_madrid.csv',
                   output_path: str = 'price_prediction_model.pkl') -> Dict:
    """
    Complete training pipeline.

    Parameters:
    -----------
    csv_path : str
        Path to input CSV file
    output_path : str
        Path to save trained model

    Returns:
    --------
    dict
        Model data dictionary
    """
    print("Loading and preparing data...")
    X, y, feature_columns, imputer = load_and_prepare_data(csv_path)
    print(f"Dataset shape: {X.shape}")

    print("\nTraining model...")
    model, metrics = train_and_evaluate(X, y)

    print("\nModel Performance:")
    print(f"  MAE:  €{metrics['test_mae']:,.2f}")
    print(f"  RMSE: €{metrics['test_rmse']:,.2f}")
    print(f"  R²:   {metrics['test_r2']:.4f}")

    print(f"\nSaving model to '{output_path}'...")
    save_model(model, imputer, feature_columns, metrics, output_path)
    print("Done!")

    return {
        'model': model,
        'imputer': imputer,
        'feature_columns': feature_columns,
        'metrics': metrics
    }


if __name__ == "__main__":
    train_pipeline()
