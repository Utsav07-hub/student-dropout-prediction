"""Model training utilities."""

from __future__ import annotations

import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def train_models(X_train, y_train) -> dict:
    """Train baseline models and return them."""
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def tune_random_forest(X_train, y_train):
    """Tune a Random Forest classifier with GridSearchCV."""
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    # param_grid = {
    #     "n_estimators": [200, 400, 600, 800],
    #     "max_depth": [15, 25, 35, None],
    #     "min_samples_split": [2, 5, 8],
    #     "min_samples_leaf": [1, 2, 3],
    #     "max_features": ["sqrt", "log2"],
    #     "bootstrap": [True, False]
    # }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    return best_model
