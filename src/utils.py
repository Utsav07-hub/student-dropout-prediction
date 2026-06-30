"""Utility helpers."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def save_model(model: Any, file_path: str | Path) -> None:
    """Save a trained model to disk."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(file_path: str | Path) -> Any:
    """Load a trained model from disk."""
    return joblib.load(file_path)


def save_csv(data: pd.DataFrame | pd.Series, file_path: str | Path, index: bool = False) -> None:
    """Save a pandas object as a CSV file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=index)
