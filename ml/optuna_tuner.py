"""
Bayesian hyperparameter tuning with Optuna — search LightGBM hyperparameters
using TIME-AWARE validation (no random K-fold leakage).

The search objective is the mean OOS daily IC across multiple walk-forward
folds. This optimizes for what actually matters in trading: out-of-sample
ranking accuracy, not in-sample fit.

NOTE: Even after careful validation, hyperparameter tuning can overfit to the
specific train/test boundaries. We use 3 folds of expanding-window WF and a
modest number of trials (30) to limit this risk.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_walk_forward_splits(dates: pd.DatetimeIndex, n_folds: int = 3) -> list:
    """
    Create expanding-window WF splits over the available date range.

    Returns list of (train_end, test_start, test_end) tuples.
    """
    unique_dates = pd.DatetimeIndex(dates).unique().sort_values()
    n_total = len(unique_dates)
    # Reserve first 50% for initial training, then split the rest into n_folds test windows
    initial_train_end_idx = n_total // 2
    test_pool_size = n_total - initial_train_end_idx
    fold_size = test_pool_size // n_folds

    folds = []
    for k in range(n_folds):
        train_end_idx = initial_train_end_idx + k * fold_size
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + fold_size, n_total - 1)
        if test_end_idx > test_start_idx:
            folds.append((unique_dates[train_end_idx],
                          unique_dates[test_start_idx],
                          unique_dates[test_end_idx]))
    return folds


def daily_rank_ic(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Mean per-date Spearman correlation between predictions and true ranks."""
    df = pd.DataFrame({'y': y_true.values, 'p': y_pred}, index=y_true.index)
    daily = df.groupby(level='date').apply(
        lambda g: g['y'].corr(g['p'], method='spearman') if len(g) > 5 else np.nan
    ).dropna()
    return float(daily.mean()) if len(daily) > 0 else 0.0


def evaluate_params_wf(params: dict, X: pd.DataFrame, y: pd.Series,
                       folds: list) -> float:
    """Train LGBM with given params on each fold, return mean OOS IC."""
    dates = X.index.get_level_values('date')
    ics = []
    for train_end, test_start, test_end in folds:
        train_mask = dates < train_end
        test_mask = (dates >= test_start) & (dates <= test_end)
        if train_mask.sum() < 5000 or test_mask.sum() < 500:
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]

        model = lgb.LGBMRegressor(
            n_estimators=params.pop('n_estimators', 300),
            **params,
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        ic = daily_rank_ic(y_te, pred)
        ics.append(ic)

    return float(np.mean(ics)) if ics else 0.0


def make_objective(X: pd.DataFrame, y: pd.Series, folds: list,
                    base_params: dict = None):
    """Build an Optuna objective function over LightGBM hyperparameters."""
    base_params = base_params or {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': -1,
    }

    def objective(trial):
        params = dict(base_params)
        params.update({
            'n_estimators': trial.suggest_int('n_estimators', 150, 500, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.06, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 30, 200),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
        })

        ic = evaluate_params_wf(params, X, y, folds)
        return ic  # maximize mean OOS IC

    return objective


def tune_lgbm(X: pd.DataFrame, y: pd.Series, n_trials: int = 30,
              n_folds: int = 3, seed: int = 42) -> dict:
    """
    Run Optuna search; return the best params + their CV score.
    """
    dates = X.index.get_level_values('date')
    folds = make_walk_forward_splits(dates, n_folds=n_folds)
    print(f"  Walk-forward folds: {len(folds)}")
    for f in folds:
        print(f"    train < {f[0].date()}, test [{f[1].date()} .. {f[2].date()}]")

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(make_objective(X, y, folds), n_trials=n_trials, show_progress_bar=False)

    return {
        'best_params': study.best_params,
        'best_ic': study.best_value,
        'n_trials': n_trials,
        'study': study,
    }
