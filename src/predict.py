"""Prediction utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import load_model


def predict(new_data, model=None, model_path: str | Path | None = None, preprocessor=None, label_encoder=None):
    """Load a model and predict for a new student record."""
    if model is None:
        if model_path is None:
            raise ValueError("Either a trained model or model_path must be provided.")
        model = load_model(model_path)

    if isinstance(new_data, dict):
        new_data = pd.DataFrame([new_data])
    elif isinstance(new_data, pd.Series):
        new_data = pd.DataFrame([new_data])
    elif not isinstance(new_data, pd.DataFrame):
        new_data = pd.DataFrame(new_data)

    if preprocessor is not None:
        transformed_data = preprocessor.transform(new_data)
    else:
        transformed_data = new_data

    prediction = model.predict(transformed_data)

    if label_encoder is not None:
        return label_encoder.inverse_transform(prediction)

    return prediction
