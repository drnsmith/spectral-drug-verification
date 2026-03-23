"""
Load spectral dataset and prepare matrices for modelling.
"""

from pathlib import Path
import pandas as pd


LABEL_COL = "label"


def load_spectral_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a spectral dataset from CSV.

    Expected structure:
    wavelength_1 | wavelength_2 | ... | wavelength_n | label

    If no column named 'label' exists, the last column is used and renamed.
    """
    file_path = Path(file_path)
    df = pd.read_csv(file_path, encoding='utf-8')

    if LABEL_COL not in df.columns:
        last_col = df.columns[-1]
        df = df.rename(columns={last_col: LABEL_COL})

    return df


def split_features_and_target(df: pd.DataFrame):
    """
    Split dataframe into spectral feature matrix and labels.
    """
    X_df = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL].values
    return X_df, y


def get_ml_matrices(df: pd.DataFrame):
    """
    Return X and y for ML tasks.
    """
    X_df, y = split_features_and_target(df)
    X = X_df.values
    return X, y


def get_wavelengths(df: pd.DataFrame):
    """
    Return wavelength values parsed from spectral column names.
    """
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    wavelengths = [float(c) for c in feature_cols]
    return wavelengths
