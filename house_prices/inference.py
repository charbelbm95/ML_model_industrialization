import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from house_prices.preprocess import preprocess_data
from house_prices import NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURES_LIST


def transform_data(df: pd.DataFrame,
                   numeric_features: list,
                   categorical_features: list,
                   scaler: StandardScaler,
                   encoder: OneHotEncoder) -> pd.DataFrame:
    """Transform the data using the provided scaler and encoder."""
    df = df.copy()
    df.loc[:, numeric_features] = scaler.transform(df[numeric_features])
    encoded_array = encoder.transform(df[categorical_features])
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=[
            f'{feature}_is_{category}'
            for feature,
            categories in zip(categorical_features, encoder.categories_)
            for category in categories],
        index=df.index)
    df = pd.concat([df.drop(categorical_features, axis=1), encoded_df], axis=1)
    return df


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    """Make predictions using the trained model."""
    loaded_scaler = joblib.load('../models/scaler.joblib')
    loaded_encoder = joblib.load('../models/ohe_encoder.joblib')
    loaded_model = joblib.load('../models/linear_regression_model.joblib')
    X_testing = input_data.copy()[FEATURES_LIST]
    X_testing = preprocess_data(X_testing,
                                NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    X_testing = transform_data(
        X_testing, NUMERIC_FEATURES,
        CATEGORICAL_FEATURES, loaded_scaler, loaded_encoder
    )
    return loaded_model.predict(X_testing)
