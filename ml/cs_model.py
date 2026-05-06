"""
Cross-sectional Ranking Model — single LightGBM regressor on the full panel.

Phase 1 trained one model per stock (per-stock specificity, but no information
sharing). Phase 2 trains one model on the entire panel with cross-sectional
features so the model learns the relative-strength signal directly.

The target is the percentile rank of forward 5-day return within each date.
The model regresses on it; predictions are ranks we use to select top-K.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from typing import Tuple


CS_LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.02,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 100,   # Larger min: lots of panel data, avoid overfit
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbosity': -1,
    'random_state': 42,
}
CS_N_ESTIMATORS = 400


def assemble_panel_features(
    cs_panel: pd.DataFrame,
    market_context: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine cross-sectional panel features with market-context features.

    Args:
        cs_panel: DataFrame indexed by (date, symbol), per-stock + cs_* features.
        market_context: DataFrame indexed by date, market-wide features.

    Returns:
        Panel DataFrame with all features broadcast to (date, symbol).
    """
    out = cs_panel.copy()
    # Broadcast market context to each (date, symbol) row
    mc = market_context.copy()
    mc.index.name = 'date'
    out = out.join(mc, on='date', how='inner')
    return out.dropna()


def align_panel_features_labels(
    panel_features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_lag: int = 1,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Lag features by `feature_lag` days so end-of-t features predict t+1 entry.

    Args:
        panel_features: MultiIndex (date, symbol) DataFrame.
        labels: MultiIndex (date, symbol) DataFrame with 'rank_label' column.
        feature_lag: Days to lag features within each symbol.

    Returns:
        (X, y) aligned by (date, symbol).
    """
    # Lag within each symbol
    lagged = (
        panel_features
        .groupby(level='symbol')
        .shift(feature_lag)
    )
    aligned = lagged.join(labels, how='inner').dropna()
    y = aligned['rank_label']
    X = aligned.drop(columns=['rank_label'])
    return X, y


def time_split_panel(X: pd.DataFrame, y: pd.Series, train_through: pd.Timestamp):
    """Split a panel by date: training = entries with date <= train_through."""
    dates = X.index.get_level_values('date')
    train_mask = dates <= train_through
    return X[train_mask], y[train_mask], X[~train_mask], y[~train_mask]


def train_cs_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict = None,
    n_estimators: int = CS_N_ESTIMATORS,
) -> lgb.LGBMRegressor:
    """Train a fresh LightGBM regressor on the panel."""
    p = {**CS_LGBM_PARAMS, **(params or {})}
    model = lgb.LGBMRegressor(n_estimators=n_estimators, **p)
    model.fit(X, y)
    return model


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Evaluate predictions on a panel.

    Key metric is rank IC: per-date Spearman corr between predicted rank and
    forward return rank. A daily IC > 0.03 is solid for equity ranking.
    """
    df = pd.concat([y_true.rename('true'), pd.Series(y_pred, index=y_true.index, name='pred')], axis=1)
    df = df.dropna()

    # Per-date rank IC
    daily_ic = df.groupby(level='date').apply(
        lambda g: g['pred'].corr(g['true'], method='spearman')
        if len(g) > 5 else np.nan
    )
    daily_ic = daily_ic.dropna()

    rmse = float(np.sqrt(mean_squared_error(df['true'], df['pred'])))
    overall_corr = float(df['pred'].corr(df['true'], method='spearman'))

    return {
        'mean_daily_ic': float(daily_ic.mean()),
        'std_daily_ic':  float(daily_ic.std()),
        'ic_t_stat':     float(daily_ic.mean() / (daily_ic.std() / np.sqrt(len(daily_ic)))) if daily_ic.std() > 0 else 0.0,
        'positive_ic_days': float((daily_ic > 0).mean()),
        'overall_spearman':  overall_corr,
        'rmse': rmse,
        'n_days': len(daily_ic),
        'n_predictions': len(df),
    }


def feature_importance(model: lgb.LGBMRegressor, feature_names: list) -> pd.Series:
    """Gain-based feature importance from a trained LGBM."""
    return pd.Series(
        model.booster_.feature_importance(importance_type='gain'),
        index=feature_names,
    ).sort_values(ascending=False)
