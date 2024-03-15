import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(df: pd.DataFrame,
                    numeric_features: list,
                    categorical_features: list) -> pd.DataFrame:
    """Preprocess the data by filling missing values."""
    df_processed = df.copy()
    df_processed[numeric_features] = df_processed[numeric_features].fillna(
        df_processed[numeric_features].mean())
    for col in categorical_features:
        df_processed[col] = df_processed[col].fillna(
            df_processed[col].mode()[0])
    return df_processed


def standardize_data(df: pd.DataFrame,
                     numeric_features: list
                     ) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize numerical features."""
    scaler = StandardScaler()
    scaler.fit(df[numeric_features])
    df[numeric_features] = scaler.transform(df[numeric_features])
    return df, scaler


def encode_features(df: pd.DataFrame,
                    features: list) -> tuple[pd.DataFrame, OneHotEncoder]:
    """Encode categorical features."""
    encoder = OneHotEncoder(sparse_output=False, dtype=np.int64)
    encoder.fit(df[features])
    encoded_df = encoder.transform(df[features])
    columns = []
    for feature, categories in zip(features, encoder.categories_):
        columns.extend([f'{feature}_is_{category}' for category in categories])
    encoded_df = pd.DataFrame(encoded_df, columns=columns, index=df.index)
    df = df.drop(features, axis=1)
    df = pd.concat([df, encoded_df], axis=1)
    return df, encoder
