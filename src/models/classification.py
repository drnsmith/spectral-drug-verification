import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_test_classification(
    X_df: pd.DataFrame,
    y,
    test_size: float = 0.2,
    random_seed: int = 42,
):
    """
    Train a simple classifier and return fitted model plus evaluation outputs.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": acc,
        "report": report,
    }