import pandas as pd
import numpy as np

from src.preprocessing.spectral_preprocessing import preprocess_spectra


def test_preprocess_spectra_returns_dataframe_and_wavelengths():
    cols = [str(float(x)) for x in range(150, 301)]
    X_df = pd.DataFrame(np.random.rand(5, len(cols)), columns=cols)

    X_processed_df, wavelengths = preprocess_spectra(
        X_df,
        crop_min=180,
        crop_max=250,
        apply_snv=True,
        apply_derivative=True,
    )

    assert isinstance(X_processed_df, pd.DataFrame)
    assert len(wavelengths) == X_processed_df.shape[1]
    assert X_processed_df.shape[0] == 5
    assert X_processed_df.isna().sum().sum() == 0