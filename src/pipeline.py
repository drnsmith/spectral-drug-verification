from pathlib import Path
import yaml

from src.ingestion.load_data import (
    load_spectral_dataset,
    split_features_and_target,
)
from src.preprocessing.spectral_preprocessing import preprocess_spectra
from src.models.classification import train_test_classification


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_directories(config: dict) -> None:
    Path(config["paths"]["processed_data_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["figures_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["reports_dir"]).mkdir(parents=True, exist_ok=True)


def run_pipeline(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    ensure_directories(config)

    df = load_spectral_dataset(config["paths"]["raw_data"])
    X_df, y = split_features_and_target(df)

    X_processed_df, wavelengths = preprocess_spectra(
        X_df,
        crop_min=config["preprocessing"]["crop_min"],
        crop_max=config["preprocessing"]["crop_max"],
        apply_snv=config["preprocessing"]["apply_snv"],
        apply_derivative=config["preprocessing"]["apply_derivative"],
    )

    clf_results = train_test_classification(
        X_processed_df,
        y,
        test_size=config["classification"]["test_size"],
        random_seed=config["random_seed"],
    )

    print("Pipeline ran successfully.")
    print(f"Raw dataframe shape: {df.shape}")
    print(f"Processed feature matrix shape: {X_processed_df.shape}")
    print(f"Number of labels: {len(set(y))}")
    print(f"Processed wavelength range: {wavelengths[0]} to {wavelengths[-1]}")
    print(f"Classification accuracy: {clf_results['accuracy']:.4f}")

    report_path = Path(config["paths"]["reports_dir"]) / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Classification accuracy: {clf_results['accuracy']:.4f}\n\n")
        f.write(clf_results["report"])

    print(f"Saved classification report to: {report_path}")

    return {
        "df": df,
        "X_processed_df": X_processed_df,
        "y": y,
        "wavelengths": wavelengths,
        "clf_results": clf_results,
    }