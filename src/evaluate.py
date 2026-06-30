"""Model evaluation utilities."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score


def evaluate_model(model, X_train, X_test, y_train, y_test) -> dict:
    """Evaluate a trained model and return common metrics."""
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "cv_score": cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean(),
        "classification_report": classification_report(y_test, y_pred),
    }
