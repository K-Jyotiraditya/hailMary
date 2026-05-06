"""
ML Preprocessor — Build aligned (X, y, t1) datasets from features and labels.

Critical to align features to entry timestamps and ensure features available
at decision time use only PAST data. Features are computed at the close of
day t and used to predict the triple-barrier label initiated at t+1.
"""
import pandas as pd
import numpy as np
from typing import Tuple


def align_features_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_lag: int = 1,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Align features (lagged) to triple-barrier labels.

    Args:
        features: DataFrame indexed by date, columns = features.
                  These are computed using data through end-of-day t.
        labels: DataFrame from triple_barrier_labels(), indexed by entry
                timestamp with columns 'bin' (label) and 't1' (exit time).
        feature_lag: Lag features by this many days before joining to labels.
                     Default 1: features at end-of-t predict label entered t+1.

    Returns:
        (X, y, t1) aligned on the same index. X has features, y has labels
        in {0, 1} (binary classification: -1/0 collapsed to 0), t1 is the
        label-horizon end timestamp for purging.
    """
    # Lag features so end-of-t features predict t+1 entry
    X_lagged = features.shift(feature_lag)

    # Inner-join on date
    aligned = X_lagged.join(labels[['bin', 't1']], how='inner').dropna()

    feature_cols = [c for c in aligned.columns if c not in ('bin', 't1')]
    X = aligned[feature_cols]

    # Convert label to binary: positive return = 1, else 0
    # (Drops the '0' time-barrier neutrals into the negative class for binary CV)
    y = (aligned['bin'] > 0).astype(int)
    t1 = aligned['t1']

    return X, y, t1


def time_split(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    train_end: str,
    val_end: str,
) -> dict:
    """
    Three-way time-ordered split: train -> validation -> test.

    Args:
        train_end: Last date (inclusive) for training set.
        val_end:   Last date (inclusive) for validation set.

    Returns:
        Dict with X_train/y_train/t1_train, X_val/y_val/t1_val, X_test/y_test/t1_test.
    """
    train_mask = X.index <= train_end
    val_mask = (X.index > train_end) & (X.index <= val_end)
    test_mask = X.index > val_end

    return {
        'X_train': X[train_mask], 'y_train': y[train_mask], 't1_train': t1[train_mask],
        'X_val':   X[val_mask],   'y_val':   y[val_mask],   't1_val':   t1[val_mask],
        'X_test':  X[test_mask],  'y_test':  y[test_mask],  't1_test':  t1[test_mask],
    }


def stack_panel(per_symbol: dict) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Stack per-symbol (X, y, t1) into a single panel with symbol identifier.

    Args:
        per_symbol: dict {symbol: (X, y, t1)}

    Returns:
        (X, y, t1, symbol_series) all aligned by row.
    """
    X_list, y_list, t1_list, sym_list = [], [], [], []

    for symbol, (X, y, t1) in per_symbol.items():
        X_list.append(X)
        y_list.append(y)
        t1_list.append(t1)
        sym_list.append(pd.Series([symbol] * len(X), index=X.index))

    X_all = pd.concat(X_list, axis=0)
    y_all = pd.concat(y_list, axis=0)
    t1_all = pd.concat(t1_list, axis=0)
    sym_all = pd.concat(sym_list, axis=0)

    # Sort by date so CPCV groups are time-contiguous (critical for purging)
    order = X_all.index.argsort()
    return X_all.iloc[order], y_all.iloc[order], t1_all.iloc[order], sym_all.iloc[order]
