"""
Multi-Horizon Model — train separate ranking models for 1d, 5d, 20d forward
returns, then ensemble their predictions.

THE INTUITION:
  Different horizons capture different alphas:
    - 1-day: short-term mean-reversion, microstructure noise — high turnover
    - 5-day: trend-continuation, sentiment shifts — our default
    - 20-day: monthly momentum, macro factors — slow and stable

  Each horizon has its own decay profile. Models trained on different
  horizons make less-correlated mistakes than a single horizon. Ensembling
  smooths out regime-specific failures (e.g., 5d signal stops working but
  20d momentum keeps producing alpha).

NOTE on transaction costs: more horizons doesn't mean we trade more often.
The portfolio still rebalances monthly. The horizons just inform what to hold.
"""
import numpy as np
import pandas as pd
from typing import Tuple

from labels.rank_label import build_label_panel


HORIZONS = [1, 5, 20]  # forward-return horizons in trading days


def build_multi_horizon_labels(prices_dir: str = 'data/processed',
                                horizons=HORIZONS) -> dict:
    """
    For each horizon, build a percentile-rank cross-sectional label panel.

    Returns: {horizon: DataFrame indexed by (date, symbol) with column 'rank_label'}
    """
    out = {}
    for h in horizons:
        out[h] = build_label_panel(horizon=h, method='pct_rank')
    return out


def train_horizon_models(X: pd.DataFrame, label_panels: dict,
                         model_factory, train_through: pd.Timestamp = None) -> dict:
    """
    Train one model per horizon.

    Args:
        X: Panel features, MultiIndex (date, symbol).
        label_panels: dict {horizon: label DataFrame}.
        model_factory: Callable() -> fresh trainable model with .fit/.predict.
        train_through: If set, restrict to X rows with date < train_through.

    Returns: {horizon: trained model}
    """
    if train_through is not None:
        dates = X.index.get_level_values('date')
        X = X[dates < train_through]

    models = {}
    for horizon, labels in label_panels.items():
        # Lag features by 1 day per stock so end-of-t features predict t+1 entry
        # (We assume X is already lagged appropriately by the caller)
        aligned = X.join(labels, how='inner').dropna()
        if len(aligned) < 5000:
            continue
        X_tr = aligned.drop(columns=['rank_label'])
        y_tr = aligned['rank_label']
        model = model_factory()
        model.fit(X_tr, y_tr)
        models[horizon] = model
    return models


def predict_horizon_ensemble(models: dict, X: pd.DataFrame,
                              weights: dict = None) -> pd.Series:
    """
    Average predictions across horizons (optionally weighted), then re-rank.

    Args:
        models: {horizon: trained model}
        X: features to predict on
        weights: {horizon: float} blend weights (default: uniform).

    Returns:
        Series of predictions indexed by X's index, re-ranked cross-sectionally.
    """
    if weights is None:
        weights = {h: 1.0 / len(models) for h in models}

    preds = pd.Series(0.0, index=X.index)
    for horizon, model in models.items():
        w = weights.get(horizon, 0.0)
        if w == 0:
            continue
        preds = preds + w * pd.Series(model.predict(X), index=X.index)

    # Cross-sectional re-rank to keep predictions on a comparable scale
    return preds.groupby(level='date').rank(pct=True)


def compute_horizon_weights_by_recent_ic(models: dict, X_recent: pd.DataFrame,
                                          labels_recent: dict,
                                          floor: float = 0.0) -> dict:
    """
    Set blend weights proportional to each horizon's recent OOS IC.
    Horizons with negative recent IC get weight = `floor` (default 0).
    """
    ics = {}
    for h, model in models.items():
        labels = labels_recent.get(h)
        if labels is None:
            continue
        aligned = X_recent.join(labels, how='inner').dropna()
        if len(aligned) < 100:
            continue
        X_a = aligned.drop(columns=['rank_label'])
        y_a = aligned['rank_label']
        pred = pd.Series(model.predict(X_a), index=X_a.index)
        df = pd.concat([y_a.rename('y'), pred.rename('p')], axis=1)
        daily = df.groupby(level='date').apply(
            lambda g: g['y'].corr(g['p'], method='spearman')
            if len(g) > 5 else np.nan
        ).dropna()
        ics[h] = max(float(daily.mean()) if len(daily) > 0 else 0.0, floor)

    total = sum(ics.values())
    if total <= 0:
        return {h: 1.0 / len(models) for h in models}
    return {h: ic / total for h, ic in ics.items()}
