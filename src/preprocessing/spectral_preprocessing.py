import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def crop_spectra(X_df: pd.DataFrame, crop_min: float, crop_max: float) -> pd.DataFrame:
    """
    Keep only spectral columns within the requested wavelength range.
    Assumes spectral columns are numeric strings, e.g. '150.0', '151.0', ...
    """
    spectral_cols = [c for c in X_df.columns if crop_min <= float(c) <= crop_max]
    return X_df[spectral_cols].copy()


def snv_normalize(X: np.ndarray) -> np.ndarray:
    """
    Standard Normal Variate normalization, row-wise.
    """
    row_means = X.mean(axis=1, keepdims=True)
    row_stds = X.std(axis=1, keepdims=True)
    row_stds[row_stds == 0] = 1.0
    return (X - row_means) / row_stds


def second_derivative(X: np.ndarray, window_length: int = 11, polyorder: int = 2) -> np.ndarray:
    """
    Apply Savitzky-Golay second derivative row-wise.
    """
    if window_length % 2 == 0:
        window_length += 1
    if window_length >= X.shape[1]:
        window_length = X.shape[1] - 1 if X.shape[1] % 2 == 0 else X.shape[1]
    if window_length < 5:
        return X.copy()

    return savgol_filter(
        X,
        window_length=window_length,
        polyorder=polyorder,
        deriv=2,
        axis=1
    )


def preprocess_spectra(
    X_df: pd.DataFrame,
    crop_min: float = 400,
    crop_max: float = 1800,
    apply_snv: bool = True,
    apply_derivative: bool = True
):
    """
    Minimal preprocessing pipeline for spectral ML.
    Returns:
        X_processed_df: processed spectra as DataFrame
        wavelengths: list of wavelength floats
    """
    X_crop = crop_spectra(X_df, crop_min=crop_min, crop_max=crop_max)
    wavelengths = [float(c) for c in X_crop.columns]

    X = X_crop.values.astype(float)

    if apply_snv:
        X = snv_normalize(X)

    if apply_derivative:
        X = second_derivative(X)

    X_processed_df = pd.DataFrame(X, columns=X_crop.columns, index=X_crop.index)
    return X_processed_df, wavelengths