"""Data preprocessing utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.features import FEATURE_COLUMNS


def preprocess_data(df: pd.DataFrame, target_column: str = "Target") -> dict:
    """Prepare features and labels for model training."""
    working_df = df.copy()

    if target_column not in working_df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataframe.")

    missing_features = [column for column in FEATURE_COLUMNS if column not in working_df.columns]
    if missing_features:
        raise KeyError(f"Missing feature columns: {missing_features}")

    X = working_df[FEATURE_COLUMNS]
    y = working_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
    }
