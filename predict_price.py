"""
Price Prediction Module for Madrid Houses

Core functions to load the trained model and make price predictions.
Can be imported in notebooks or other scripts.

Example usage:
    from predict_price import load_model, predict_single, predict_batch

    model_data = load_model('price_prediction_model.pkl')
    price = predict_single(model_data, sq_mt_built=85, n_rooms=3, n_bathrooms=2)
"""

import pickle
import pandas as pd
from typing import Dict, Union, Optional


def load_model(model_path: str = 'price_prediction_model.pkl') -> Dict:
    """
    Load the trained model from a pickle file.

    Parameters:
    -----------
    model_path : str
        Path to the pickle file containing the model

    Returns:
    --------
    dict
        Dictionary containing model, imputer, feature_columns, and metadata
    """
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def get_required_features(model_data: Dict) -> list:
    """
    Get list of required features for the model.

    Parameters:
    -----------
    model_data : dict
        Dictionary returned by load_model()

    Returns:
    --------
    list
        List of required feature column names
    """
    return model_data['feature_columns']


def check_dataframe_features(df: pd.DataFrame, model_data: Dict) -> Dict:
    """
    Check which required features are missing from the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    model_data : dict
        Dictionary returned by load_model()

    Returns:
    --------
    dict
        Dictionary with 'missing' and 'present' feature lists
    """
    required_features = model_data['feature_columns']
    df_columns = set(df.columns)

    missing = [col for col in required_features if col not in df_columns]
    present = [col for col in required_features if col in df_columns]

    return {
        'missing': missing,
        'present': present,
        'total_required': len(required_features)
    }


def predict_single(model_data: Dict,
                   sq_mt_built: float,
                   n_rooms: int,
                   n_bathrooms: int,
                   sq_mt_useful: Optional[float] = None,
                   n_floors: Optional[float] = None,
                   built_year: Optional[float] = None,
                   has_central_heating: int = 0,
                   has_individual_heating: int = 0,
                   has_ac: int = 0,
                   has_fitted_wardrobes: int = 0,
                   has_lift: int = 0,
                   is_exterior: int = 0,
                   has_garden: int = 0,
                   has_pool: int = 0,
                   has_terrace: int = 0,
                   has_balcony: int = 0,
                   has_storage_room: int = 0,
                   is_accessible: int = 0,
                   has_green_zones: int = 0,
                   has_parking: int = 0,
                   is_parking_included_in_price: int = 0,
                   parking_price: float = 0.0,
                   is_orientation_north: int = 0,
                   is_orientation_west: int = 0,
                   is_orientation_south: int = 0,
                   is_orientation_east: int = 0,
                   is_renewal_needed: int = 0,
                   is_new_development: int = 0) -> float:
    """
    Predict price for a single property.

    Parameters:
    -----------
    model_data : dict
        Dictionary returned by load_model()
    sq_mt_built : float
        Built area in square meters (required)
    n_rooms : int
        Number of rooms (required)
    n_bathrooms : int
        Number of bathrooms (required)
    **other parameters : optional property features

    Returns:
    --------
    float
        Predicted price in euros
    """
    property_dict = {
        'sq_mt_built': sq_mt_built,
        'sq_mt_useful': sq_mt_useful if sq_mt_useful is not None else sq_mt_built * 0.85,
        'n_rooms': n_rooms,
        'n_bathrooms': n_bathrooms,
        'n_floors': n_floors,
        'built_year': built_year,
        'has_central_heating': has_central_heating,
        'has_individual_heating': has_individual_heating,
        'has_ac': has_ac,
        'has_fitted_wardrobes': has_fitted_wardrobes,
        'has_lift': has_lift,
        'is_exterior': is_exterior,
        'has_garden': has_garden,
        'has_pool': has_pool,
        'has_terrace': has_terrace,
        'has_balcony': has_balcony,
        'has_storage_room': has_storage_room,
        'is_accessible': is_accessible,
        'has_green_zones': has_green_zones,
        'has_parking': has_parking,
        'is_parking_included_in_price': is_parking_included_in_price,
        'parking_price': parking_price,
        'is_orientation_north': is_orientation_north,
        'is_orientation_west': is_orientation_west,
        'is_orientation_south': is_orientation_south,
        'is_orientation_east': is_orientation_east,
        'is_renewal_needed': is_renewal_needed,
        'is_new_development': is_new_development
    }

    model = model_data['model']
    imputer = model_data['imputer']
    feature_columns = model_data['feature_columns']

    X_new = pd.DataFrame([property_dict])[feature_columns]
    X_new_imputed = pd.DataFrame(
        imputer.transform(X_new),
        columns=feature_columns
    )

    return model.predict(X_new_imputed)[0]


def predict_batch(model_data: Dict,
                  df: pd.DataFrame,
                  add_to_dataframe: bool = True) -> Union[pd.Series, pd.DataFrame]:
    """
    Predict prices for multiple properties from a DataFrame.

    Parameters:
    -----------
    model_data : dict
        Dictionary returned by load_model()
    df : pd.DataFrame
        DataFrame containing property features
    add_to_dataframe : bool, default=True
        If True, adds 'predicted_price' column to DataFrame
        If False, returns only predictions as Series

    Returns:
    --------
    pd.DataFrame or pd.Series
        DataFrame with predictions or Series of predictions
    """
    model = model_data['model']
    imputer = model_data['imputer']
    feature_columns = model_data['feature_columns']

    X_new = df[feature_columns].copy()
    X_new_imputed = pd.DataFrame(
        imputer.transform(X_new),
        columns=feature_columns,
        index=X_new.index
    )

    predictions = model.predict(X_new_imputed)

    if add_to_dataframe:
        df = df.copy()
        df['predicted_price'] = predictions
        return df
    else:
        return pd.Series(predictions, index=df.index)


def predict_from_csv(model_data: Dict,
                     csv_path: str,
                     output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load properties from CSV and predict prices.

    Parameters:
    -----------
    model_data : dict
        Dictionary returned by load_model()
    csv_path : str
        Path to CSV file containing properties
    output_path : str, optional
        If provided, saves results to this path

    Returns:
    --------
    pd.DataFrame
        DataFrame with predicted prices
    """
    df = pd.read_csv(csv_path)
    df_with_predictions = predict_batch(model_data, df, add_to_dataframe=True)

    if output_path:
        df_with_predictions.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")

    return df_with_predictions
