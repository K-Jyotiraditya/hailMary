"""
Walk-Forward Analysis — quantify in-sample vs out-of-sample degradation.

For each fold:
  - Train on data up to time T
  - Test on data [T, T + window]
  - Compute IS Sharpe (training-period strategy returns) and OOS Sharpe
  - Track degradation = (IS - OOS) / IS

Aggregate metrics across folds:
  - Mean OOS Sharpe — the realistic expectation
  - Mean degradation — how much performance drops out-of-sample
  - Stability — standard deviation of OOS Sharpe across folds

CRITERIA FOR ROBUSTNESS:
  - Mean OOS Sharpe > 0.3 (some real edge)
  - Degradation < 30% (model generalizes)
  - OOS Sharpe std < 0.5 (consistent across regimes)
"""
import numpy as np
import pandas as pd
from typing import Callable, List


def sharpe_of_returns(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Standard annualized Sharpe."""
    if len(returns) < 5 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def split_into_folds(dates: pd.DatetimeIndex, n_folds: int = 4) -> List[tuple]:
    """
    Generate (train_end, test_end) pairs for walk-forward folds.

    Each fold uses a growing training window and a fixed-size test window.
    Test windows are non-overlapping, contiguous slices of the test period.
    """
    n = len(dates)
    test_start_idx = n // 2  # First half = first training set
    test_size = (n - test_start_idx) // n_folds

    folds = []
    for k in range(n_folds):
        train_end_idx = test_start_idx + k * test_size
        test_end_idx = min(train_end_idx + test_size, n - 1)
        if test_end_idx > train_end_idx:
            folds.append((dates[train_end_idx], dates[test_end_idx]))
    return folds


def fold_metrics(returns: pd.Series, train_end: pd.Timestamp,
                 test_end: pd.Timestamp) -> dict:
    """Compute IS / OOS Sharpe for a single fold."""
    is_returns = returns[returns.index <= train_end]
    oos_returns = returns[(returns.index > train_end) & (returns.index <= test_end)]

    is_sharpe = sharpe_of_returns(is_returns)
    oos_sharpe = sharpe_of_returns(oos_returns)
    degradation = (is_sharpe - oos_sharpe) / is_sharpe if is_sharpe != 0 else np.nan

    return {
        'train_end': train_end,
        'test_end': test_end,
        'n_is': len(is_returns),
        'n_oos': len(oos_returns),
        'is_sharpe': is_sharpe,
        'oos_sharpe': oos_sharpe,
        'is_mean_ret': float(is_returns.mean() * 252),
        'oos_mean_ret': float(oos_returns.mean() * 252),
        'degradation': degradation,
    }


def walk_forward_analysis(returns: pd.Series, n_folds: int = 4) -> pd.DataFrame:
    """
    Run walk-forward analysis on a return series.

    Returns DataFrame with one row per fold.
    """
    folds = split_into_folds(returns.index, n_folds)
    rows = [fold_metrics(returns, te, oe) for te, oe in folds]
    return pd.DataFrame(rows)


def summarize_walk_forward(wf_df: pd.DataFrame) -> dict:
    """Aggregate WF results into pass/fail summary."""
    return {
        'mean_is_sharpe':  wf_df['is_sharpe'].mean(),
        'mean_oos_sharpe': wf_df['oos_sharpe'].mean(),
        'std_oos_sharpe':  wf_df['oos_sharpe'].std(),
        'mean_degradation': wf_df['degradation'].mean(),
        'positive_oos_folds': int((wf_df['oos_sharpe'] > 0).sum()),
        'n_folds': len(wf_df),
        'pass_min_oos_sharpe': bool(wf_df['oos_sharpe'].mean() > 0.3),
        'pass_degradation':    bool(wf_df['degradation'].mean() < 0.3),
        'pass_consistency':    bool(wf_df['oos_sharpe'].std() < 0.5),
    }


def bootstrap_sharpe_ci(returns: pd.Series, n_bootstrap: int = 5000,
                       confidence: float = 0.95, block_size: int = 20,
                       random_state: int = 42) -> dict:
    """
    Bootstrap confidence interval for Sharpe ratio using moving block bootstrap.

    Block bootstrap preserves serial correlation that simple bootstrap destroys
    — important for return series with autocorrelation in volatility.
    """
    rng = np.random.RandomState(random_state)
    rets = returns.dropna().values
    n = len(rets)

    if n < block_size * 5:
        return {'ci_lower': np.nan, 'ci_upper': np.nan, 'p_value': np.nan}

    sharpes = np.empty(n_bootstrap)
    n_blocks = n // block_size + 1
    for i in range(n_bootstrap):
        starts = rng.randint(0, n - block_size, size=n_blocks)
        sample_idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        sample = rets[sample_idx]
        std = sample.std()
        sharpes[i] = (sample.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    alpha = (1 - confidence) / 2
    return {
        'observed_sharpe': sharpe_of_returns(returns),
        'ci_lower':  float(np.quantile(sharpes, alpha)),
        'ci_upper':  float(np.quantile(sharpes, 1 - alpha)),
        'p_value':   float((sharpes <= 0).mean()),  # one-sided: P(Sharpe <= 0)
        'n_bootstrap': n_bootstrap,
    }
