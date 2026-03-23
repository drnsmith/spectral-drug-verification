from src.pipeline import run_pipeline
def test_run_pipeline_smoke():
    results = run_pipeline("configs/config.yaml")

    assert "df" in results
    assert "X_processed_df" in results
    assert "y" in results
    assert "clf_results" in results
    assert results["X_processed_df"].shape[0] == len(results["y"])