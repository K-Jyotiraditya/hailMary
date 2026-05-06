"""
Multi-Seed Bagging for LightGBM — train several models with different seeds,
average their predictions.

Why this beats LGBM+RF ensembling on equity panels:
  - LGBM and RF make CORRELATED mistakes (both are tree models on the same
    features). Their averaging cancels little noise.
  - Multi-seed LGBM trains the SAME architecture with different randomness
    in subsampling and feature selection. The mistakes are LESS correlated
    because they come from the same model class but different stochastic
    paths. Empirically gives a more reliable Sharpe bump than RF blending.

  Computationally cheaper than a deep stack: 5 LGBMs ~ 5x train time but
  predictions are independent and parallelizable.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from ml.cs_model import CS_LGBM_PARAMS, CS_N_ESTIMATORS


def train_bagged_lgbm(X: pd.DataFrame, y: pd.Series,
                      n_seeds: int = 5,
                      base_params: dict = None,
                      n_estimators: int = CS_N_ESTIMATORS) -> list:
    """
    Train `n_seeds` LightGBM regressors with different random seeds and
    bootstrap subsamples.

    Returns list of trained models.
    """
    base = {**CS_LGBM_PARAMS, **(base_params or {})}
    models = []

    for i in range(n_seeds):
        params = dict(base)
        params['random_state'] = 42 + i
        params['feature_fraction_seed'] = 42 + i
        params['bagging_seed'] = 42 + i
        # Slight per-seed jitter on hyperparameters for additional decorrelation
        params['colsample_bytree'] = base.get('colsample_bytree', 0.7) * (0.95 + 0.10 * (i / n_seeds))
        params['subsample'] = base.get('subsample', 0.8) * (0.95 + 0.10 * (i / n_seeds))
        params['colsample_bytree'] = float(np.clip(params['colsample_bytree'], 0.5, 1.0))
        params['subsample'] = float(np.clip(params['subsample'], 0.5, 1.0))

        model = lgb.LGBMRegressor(n_estimators=n_estimators, **params)
        model.fit(X, y)
        models.append(model)

    return models


def predict_bagged(models: list, X: pd.DataFrame) -> pd.Series:
    """
    Average predictions from a list of bagged models.

    Returns Series indexed by X's index.
    """
    preds = np.zeros(len(X))
    for m in models:
        preds += m.predict(X)
    preds /= len(models)
    return pd.Series(preds, index=X.index)


def predict_bagged_with_ci(models: list, X: pd.DataFrame) -> pd.DataFrame:
    """
    Return mean and standard deviation of predictions across the bag.
    Useful for confidence-aware position sizing or drift monitoring.
    """
    preds = np.array([m.predict(X) for m in models])  # [n_seeds, n_samples]
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return pd.DataFrame({'pred_mean': mean, 'pred_std': std}, index=X.index)
