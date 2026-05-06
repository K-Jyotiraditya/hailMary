"""
Random Forest cross-sectional regressor — same target as LightGBM (rank label).

Why bother: RF and gradient-boosted trees disagree on different parts of
feature space. Averaging them reduces variance without adding much bias —
typically a small but consistent Sharpe bump.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 12,
    'min_samples_split': 100,
    'min_samples_leaf': 50,
    'max_features': 0.5,
    'n_jobs': -1,
    'random_state': 42,
}


def train_rf(X: pd.DataFrame, y: pd.Series, params: dict = None) -> RandomForestRegressor:
    """Train a fresh RF regressor on the panel."""
    p = {**RF_PARAMS, **(params or {})}
    model = RandomForestRegressor(**p)
    model.fit(X, y)
    return model


def predict_rf(model: RandomForestRegressor, X: pd.DataFrame) -> np.ndarray:
    """Predict and return as numpy array."""
    return model.predict(X)
