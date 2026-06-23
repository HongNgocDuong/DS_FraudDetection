"""
Utility functions for Random Forest App
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, Tuple


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate if dataframe is suitable for model training.
    
    Returns:
        Tuple of (is_valid, message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty"
    
    if df.shape[0] < 10:
        return False, "Dataset too small (minimum 10 rows required)"
    
    if df.shape[1] < 2:
        return False, "Dataset needs at least 2 columns (features + target)"
    
    return True, "Dataset is valid"


def get_data_summary(df: pd.DataFrame) -> Dict:
    """Get summary statistics for a dataframe."""
    return {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'missing_count': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'numeric_cols': df.select_dtypes(include=[np.number]).shape[1],
        'categorical_cols': df.select_dtypes(include=['object', 'category']).shape[1],
        'dtypes': df.dtypes.to_dict()
    }


def get_column_statistics(df: pd.DataFrame, column: str) -> Dict:
    """Get statistics for a specific column."""
    col = df[column]
    
    if pd.api.types.is_numeric_dtype(col):
        return {
            'dtype': 'numeric',
            'mean': col.mean(),
            'std': col.std(),
            'min': col.min(),
            'max': col.max(),
            'missing': col.isnull().sum()
        }
    else:
        return {
            'dtype': 'categorical',
            'unique_values': col.nunique(),
            'top_value': col.mode()[0] if len(col.mode()) > 0 else None,
            'missing': col.isnull().sum()
        }


def encode_categorical(df: pd.DataFrame, columns: list = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical columns.
    
    Returns:
        Tuple of (encoded_dataframe, encoders_dict)
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns
    
    df_encoded = df.copy()
    encoders = {}
    
    for col in columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
    
    return df_encoded, encoders


def scale_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numeric features.
    
    Returns:
        Tuple of (scaled_features, scaler)
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )
    return X_scaled, scaler


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in dataframe.
    
    Args:
        df: Input dataframe
        strategy: 'mean' for numeric columns, 'mode' for categorical
    
    Returns:
        Dataframe with missing values handled
    """
    df_clean = df.copy()
    
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown', inplace=True)
    
    return df_clean
