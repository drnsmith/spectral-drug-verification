import pandas as pd
import numpy as np

from src.models.classification import train_test_classification


def test_train_test_classification_returns_expected_keys():
    X_df = pd.DataFrame(np.random.rand(60, 20))
    y = np.array(["A"] * 30 + ["B"] * 30)

    results = train_test_classification(X_df, y, test_size=0.2, random_seed=42)

    expected_keys = {
        "model",
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "y_pred",
        "accuracy",
        "report",
    }

    assert expected_keys.issubset(results.keys())
    assert 0.0 <= results["accuracy"] <= 1.0