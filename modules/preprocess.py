import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


def preprocess_data(df: pd.DataFrame,
                    numeric_features: list,
                    categorical_features: list,
                    type_used : str) -> pd.DataFrame:
    """Preprocess the data by filling missing values."""
    df_processed = df.copy()
    if type_used == 'training':
        numeric_features_training_mean= df_processed[numeric_features].mean()
        joblib.dump(numeric_features_training_mean, '../models/numeric_features_training_mean.joblib')
        df_processed[numeric_features] = df_processed[numeric_features].fillna(numeric_features_training_mean)
        categorical_features_mode = {}
        for col in categorical_features:
            mode_value = df_processed[col].mode()[0]
            categorical_features_mode[col] = mode_value
        joblib.dump(categorical_features_mode, '../models/categorical_features_training_mode.joblib')
        
        # Fill missing categorical features with their mode
        for col in categorical_features:
            df_processed[col] = df_processed[col].fillna(categorical_features_mode[col])
        return df_processed
    elif type_used == 'inference':
        loaded_training_num_mean = joblib.load('../models/numeric_features_training_mean.joblib')
        loaded_training_cat_mode = joblib.load('../models/categorical_features_training_mode.joblib')
        df_processed[numeric_features] = df_processed[numeric_features].fillna(loaded_training_num_mean)
        for col in categorical_features:
            df_processed[col] = df_processed[col].fillna(loaded_training_cat_mode[col])
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
