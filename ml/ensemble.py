"""
Model Ensemble — combine LightGBM and Random Forest predictions.

Two blending strategies supported:
  1. Simple average: 0.5 * LGBM + 0.5 * RF.
  2. Weighted by validation IC: weight is proportional to each model's
     out-of-sample rank-IC on a held-out validation slice.

We always re-rank the ensemble output cross-sectionally per date so the
final scores are still uniform percentile ranks (good for top-K selection).
"""
import numpy as np
import pandas as pd


def average_predictions(*predictions, weights=None) -> pd.Series:
    """
    Average multiple prediction Series (each indexed by (date, symbol)).

    Args:
        *predictions: Variable number of prediction Series.
        weights: Optional list of weights (defaults to equal weighting).

    Returns:
        Combined prediction Series.
    """
    if not predictions:
        raise ValueError("Need at least one prediction")
    if weights is None:
        weights = [1.0 / len(predictions)] * len(predictions)

    df = pd.concat([p.rename(f'pred_{i}') for i, p in enumerate(predictions)], axis=1)
    weighted = df * weights
    return weighted.sum(axis=1)


def cross_sectional_rerank(predictions: pd.Series) -> pd.Series:
    """
    Rerank predictions to percentile within each date (across symbols).
    Maintains a uniform [0, 1] distribution per day for top-K selection.
    """
    return predictions.groupby(level='date').rank(pct=True)


def ensemble_predict(lgbm_preds: pd.Series, rf_preds: pd.Series,
                     w_lgbm: float = 0.6, w_rf: float = 0.4,
                     rerank: bool = True) -> pd.Series:
    """
    Default ensemble: 60% LightGBM, 40% RF, then re-rank cross-sectionally.

    LightGBM gets slightly higher weight because it tends to capture sharper
    signals; RF provides smoothing.
    """
    combined = average_predictions(lgbm_preds, rf_preds, weights=[w_lgbm, w_rf])
    if rerank:
        combined = cross_sectional_rerank(combined)
    return combined
