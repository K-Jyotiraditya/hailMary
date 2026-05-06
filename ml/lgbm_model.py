"""
LightGBM Trainer with CPCV evaluation.

Trains a LightGBM binary classifier predicting the triple-barrier label,
evaluated via Combinatorial Purged Cross-Validation.

DESIGN NOTES:
  - Binary target: 1 if profit barrier hit OR positive return at time barrier,
    0 otherwise. This converts triple-barrier into a clean classification.
  - We score by AUC (rank-based, robust to threshold choice) and the
    correlation of predicted probability with realized forward return — that's
    what really matters for trading.
  - sample_weight is uniform here, but could be set to label uniqueness
    (de Prado's recommendation for overlapping labels).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Tuple

from ml.cpcv import cpcv_split


# Default hyperparameters (sensible for tabular financial data)
DEFAULT_LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.02,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 50,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbosity': -1,
    'random_state': 42,
}


def make_lgbm_model(params: dict = None, n_estimators: int = 500):
    """Factory returning a fresh LightGBM classifier with given params."""
    p = {**DEFAULT_LGBM_PARAMS, **(params or {})}
    return lgb.LGBMClassifier(n_estimators=n_estimators, **p)


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    params: dict = None,
    n_estimators: int = 500,
    early_stopping: int = 50,
) -> lgb.LGBMClassifier:
    """Train one LightGBM model with optional early stopping on a validation set."""
    model = make_lgbm_model(params, n_estimators)

    fit_kwargs = {}
    if X_val is not None and y_val is not None:
        fit_kwargs['eval_set'] = [(X_val, y_val)]
        fit_kwargs['callbacks'] = [lgb.early_stopping(early_stopping, verbose=False)]

    model.fit(X_train, y_train, **fit_kwargs)
    return model


def cpcv_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    forward_returns: pd.Series = None,
    n_groups: int = 6,
    k_test: int = 2,
    embargo_pct: float = 0.01,
    params: dict = None,
    n_estimators: int = 500,
) -> Tuple[pd.DataFrame, list]:
    """
    Run CPCV: train fresh LightGBM on each (purged train, test) split.

    Args:
        X, y, t1: Aligned features, labels, label-horizon timestamps.
        forward_returns: Optional Series of realized forward returns aligned
                         with X. Used to compute Spearman IC of predictions.
        params: LightGBM overrides.

    Returns:
        (metrics_df, models_list)
        metrics_df: per-split AUC, accuracy, IC, train/test sizes.
        models_list: trained models (one per split).
    """
    rows = []
    models = []

    for split_idx, (train_idx, test_idx) in enumerate(
        cpcv_split(t1, n_groups, k_test, embargo_pct)
    ):
        if len(train_idx) < 100 or len(test_idx) < 20:
            continue

        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

        model = make_lgbm_model(params, n_estimators)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]

        try:
            auc = roc_auc_score(y_te, proba)
        except ValueError:
            auc = np.nan

        acc = accuracy_score(y_te, (proba > 0.5).astype(int))

        ic = np.nan
        if forward_returns is not None:
            # Use positional indexing — forward_returns is row-aligned with X
            fwd_test = forward_returns.iloc[test_idx]
            valid = fwd_test.notna().values
            if valid.sum() > 30:
                fwd_clean = fwd_test.values[valid]
                pred_clean = proba[valid]
                ic = pd.Series(pred_clean).corr(pd.Series(fwd_clean), method='spearman')

        rows.append({
            'split': split_idx,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'auc': auc,
            'accuracy': acc,
            'ic': ic,
            'pos_rate_train': y_tr.mean(),
            'pos_rate_test': y_te.mean(),
        })
        models.append(model)

    return pd.DataFrame(rows), models


def feature_importance(model: lgb.LGBMClassifier, feature_names: list,
                       importance_type: str = 'gain') -> pd.Series:
    """Extract feature importance from a trained model."""
    return pd.Series(
        model.booster_.feature_importance(importance_type=importance_type),
        index=feature_names,
        name='importance',
    ).sort_values(ascending=False)


def aggregate_importance(models: list, feature_names: list) -> pd.DataFrame:
    """Mean ± std of feature importance across all CPCV models."""
    importance_matrix = []
    for model in models:
        imp = feature_importance(model, feature_names)
        importance_matrix.append(imp)

    df = pd.DataFrame(importance_matrix)
    return pd.DataFrame({
        'mean_importance': df.mean(),
        'std_importance': df.std(),
        'consistent': df.mean() > df.std(),  # Stable across folds
    }).sort_values('mean_importance', ascending=False)
