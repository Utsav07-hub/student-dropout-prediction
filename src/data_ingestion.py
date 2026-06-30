"""Data ingestion utilities."""

from pathlib import Path

import pandas as pd


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)
