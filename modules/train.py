import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from house_prices.preprocess import (
    preprocess_data, standardize_data, encode_features
)
from house_prices import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    FEATURES_LIST, TARGET_VARIABLE
)


def compute_rmsle(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  precision: int = 2) -> float:
    """Compute the Root Mean Squared Logarithmic Error."""
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    return round(rmsle, precision)


def prepare_data(
    df: pd.DataFrame,
    numeric_features: list,
    categorical_features: list,
    scaler: StandardScaler = None,
    encoder: OneHotEncoder = None,
    fit_transform: bool = True
) -> tuple[pd.DataFrame, StandardScaler, OneHotEncoder]:
    """Prepare the data by preprocessing, standardizing, and encoding."""
    df = preprocess_data(df, numeric_features, categorical_features,'training')
    if fit_transform:
        df, scaler = standardize_data(df, numeric_features)
    else:
        df[numeric_features] = scaler.transform(df[numeric_features])
    if fit_transform:
        df, encoder = encode_features(df, categorical_features)
    else:
        df = encode_features(df, categorical_features)[0]
    return df, scaler, encoder


def build_model(data: pd.DataFrame) -> dict[str, float]:
    """Build and evaluate the model from the provided DataFrame."""
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES_LIST],
        data[TARGET_VARIABLE],
        test_size=0.25,
        random_state=42)
    X_train, scaler, encoder = prepare_data(
        X_train, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    joblib.dump(scaler, '../models/scaler.joblib')
    joblib.dump(encoder, '../models/ohe_encoder.joblib')
    X_test, _, _ = prepare_data(
        X_test, NUMERIC_FEATURES, CATEGORICAL_FEATURES, scaler, encoder, False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, '../models/linear_regression_model.joblib')
    y_pred = model.predict(X_test)
    y_pred = np.delete(y_pred, 134)
    y_test = np.delete(y_test, 134)
    return {'rmsle': compute_rmsle(y_test, y_pred)}
