from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd

from src.pipeline import run_pipeline
from src.preprocessing.spectral_preprocessing import preprocess_spectra

app = FastAPI(title="Spectral Drug Verification API")

# Load pipeline artifacts once on startup
PIPELINE_RESULTS = run_pipeline("configs/config.yaml")
MODEL = PIPELINE_RESULTS["clf_results"]["model"]

RAW_DF = PIPELINE_RESULTS["df"]
RAW_FEATURE_COLUMNS = [c for c in RAW_DF.columns if c != "label"]

PROCESSED_FEATURE_COLUMNS = PIPELINE_RESULTS["X_processed_df"].columns.tolist()


class SpectrumRequest(BaseModel):
    spectrum: List[float]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Spectral Drug Verification API is running"
    }


@app.get("/metadata")
def metadata():
    return {
        "raw_num_features": len(RAW_FEATURE_COLUMNS),
        "raw_first_feature": RAW_FEATURE_COLUMNS[0],
        "raw_last_feature": RAW_FEATURE_COLUMNS[-1],
        "processed_num_features": len(PROCESSED_FEATURE_COLUMNS),
        "processed_first_feature": PROCESSED_FEATURE_COLUMNS[0],
        "processed_last_feature": PROCESSED_FEATURE_COLUMNS[-1],
        "num_classes": len(set(PIPELINE_RESULTS["y"])),
    }


@app.post("/predict")
def predict(request: SpectrumRequest):
    if len(request.spectrum) != len(RAW_FEATURE_COLUMNS):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected raw spectrum length {len(RAW_FEATURE_COLUMNS)}, "
                f"got {len(request.spectrum)}"
            ),
        )

    # Create raw one-row dataframe with original wavelength columns
    X_raw = pd.DataFrame([request.spectrum], columns=RAW_FEATURE_COLUMNS)

    # Apply same preprocessing as training pipeline
    X_processed, _ = preprocess_spectra(
        X_raw,
        crop_min=400,
        crop_max=1800,
        apply_snv=True,
        apply_derivative=True,
    )

    prediction = MODEL.predict(X_processed)[0]

    return {
        "predicted_label": prediction
    }