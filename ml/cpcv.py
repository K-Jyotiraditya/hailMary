"""
Combinatorial Purged Cross-Validation (CPCV) — de Prado, AFML Chapter 12.

THE PROBLEM CPCV SOLVES:
  Standard K-fold CV randomly shuffles samples. In time series, this leaks
  future information into training (a sample from 2020 might end up in train
  while a 2018 sample ends up in test, even though 2018 was supposed to be
  "past"). This produces over-optimistic performance estimates.

  Time-series K-fold (expanding window) avoids leakage but only gives ONE
  path through the data. Bad luck with one fold can mislead.

CPCV's APPROACH:
  1. Split data into N contiguous groups (e.g., N=6).
  2. For each combination of k test groups (e.g., k=2), the rest is training.
     This yields C(N,k) train/test splits — many independent paths.
  3. PURGE: Remove training samples whose LABEL HORIZON overlaps with test
     groups. (A label using forward returns from day t to t+5 leaks if any
     of those days are in test.)
  4. EMBARGO: Remove training samples in a small window after each test
     group, to prevent leakage from delayed information (e.g., post-event
     drift).

  Train on the purged/embargoed training set, evaluate on the test groups,
  aggregate metrics across all C(N,k) splits.

WHY IT MATTERS:
  CPCV gives a DISTRIBUTION of OOS performance, not just a point estimate.
  If your strategy's best Sharpe is 2.0 across 15 splits but the median is
  0.3, you're probably overfitting to one favorable regime.
"""
import numpy as np
import pandas as pd
from itertools import combinations
from typing import Iterator, Tuple


def get_train_test_splits(
    n_samples: int,
    n_groups: int = 6,
    k_test: int = 2,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield (train_idx, test_idx) for every combination of k test groups out of N.

    Indices are over rows (0 to n_samples-1), assumed time-ordered.
    Returns C(n_groups, k_test) splits.
    """
    # Partition row indices into N contiguous groups
    group_size = n_samples // n_groups
    groups = []
    for g in range(n_groups):
        start = g * group_size
        end = (g + 1) * group_size if g < n_groups - 1 else n_samples
        groups.append(np.arange(start, end))

    # All combinations of k_test groups out of n_groups
    for test_groups in combinations(range(n_groups), k_test):
        test_idx = np.concatenate([groups[g] for g in test_groups])
        train_groups = [g for g in range(n_groups) if g not in test_groups]
        train_idx = np.concatenate([groups[g] for g in train_groups])
        yield train_idx, test_idx


def purge_overlapping(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    t1: pd.Series,
) -> np.ndarray:
    """
    Remove training samples whose label horizon overlaps with ANY test sample.

    A training sample at entry time t_train with label horizon ending at
    t1_train is purged if any test sample's entry time falls within
    [t_train, t1_train]. This prevents leakage when forward-looking labels
    overlap with samples used for evaluation.

    Args:
        train_idx, test_idx: Row positions into t1.index.
        t1: Series indexed by entry timestamp -> label END timestamp.

    Returns:
        Filtered training indices (purged of overlapping samples).
    """
    if t1.empty or len(test_idx) == 0:
        return train_idx

    test_entry_times = t1.index[test_idx].values  # 1D array of test entry times

    train_entry_times = t1.index[train_idx].values
    train_t1_times = t1.iloc[train_idx].values

    # For each train sample, check if any test entry falls inside its label window.
    # Vectorized: broadcast train [N,1] vs test [1,M], detect overlap.
    test_t = test_entry_times.reshape(1, -1)  # [1, M]
    train_start = train_entry_times.reshape(-1, 1)  # [N, 1]
    train_end = train_t1_times.reshape(-1, 1)        # [N, 1]

    overlap_per_pair = (test_t >= train_start) & (test_t <= train_end)
    overlap_mask = overlap_per_pair.any(axis=1)
    return train_idx[~overlap_mask]


def apply_embargo(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_samples: int,
    embargo_pct: float = 0.01,
) -> np.ndarray:
    """
    Drop training samples within `embargo_pct * n_samples` rows after each test group.

    This guards against leakage from post-test market reactions (e.g., drift
    after announcements that the model could otherwise learn from).
    """
    embargo_size = max(1, int(embargo_pct * n_samples))
    test_set = set(test_idx)

    # Find contiguous test blocks; embargo the rows after each
    embargo_idx = set()
    test_sorted = np.sort(test_idx)
    block_end = test_sorted[0]
    for i in range(1, len(test_sorted)):
        if test_sorted[i] != test_sorted[i - 1] + 1:
            for r in range(block_end + 1, min(block_end + 1 + embargo_size, n_samples)):
                embargo_idx.add(r)
            block_end = test_sorted[i]
        else:
            block_end = test_sorted[i]
    # Embargo after the final block
    for r in range(block_end + 1, min(block_end + 1 + embargo_size, n_samples)):
        embargo_idx.add(r)

    return np.array([i for i in train_idx if i not in embargo_idx])


def cpcv_split(
    t1: pd.Series,
    n_groups: int = 6,
    k_test: int = 2,
    embargo_pct: float = 0.01,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generator yielding (purged_train_idx, test_idx) for CPCV.

    Args:
        t1: Series mapping entry_ts -> exit_ts (from triple-barrier output).
            Used to determine label horizons for purging.
        n_groups: Number of contiguous groups to split data into.
        k_test: Number of groups held out for testing per split.
        embargo_pct: Fraction of total samples to embargo after test groups.

    Yields:
        (train_idx, test_idx) pairs with purging and embargo applied.
    """
    n = len(t1)
    for train_idx, test_idx in get_train_test_splits(n, n_groups, k_test):
        train_idx = purge_overlapping(train_idx, test_idx, t1)
        train_idx = apply_embargo(train_idx, test_idx, n, embargo_pct)
        yield train_idx, test_idx


def cpcv_score(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    n_groups: int = 6,
    k_test: int = 2,
    embargo_pct: float = 0.01,
    metric=None,
) -> pd.DataFrame:
    """
    Run CPCV with a fresh model per split and return per-split metrics.

    Args:
        model_factory: Callable returning a fresh model with .fit and
                       .predict_proba (sklearn-style).
        X, y: Features and labels (aligned, indexed by entry timestamp).
        t1: Label horizon series.
        metric: Function (y_true, y_pred_proba) -> float. Default is
                accuracy on argmax.

    Returns:
        DataFrame with one row per CPCV split: train_size, test_size, score.
    """
    if metric is None:
        from sklearn.metrics import accuracy_score
        metric = lambda yt, yp: accuracy_score(yt, (yp > 0.5).astype(int))

    rows = []
    for split_idx, (train_idx, test_idx) in enumerate(
        cpcv_split(t1, n_groups, k_test, embargo_pct)
    ):
        if len(train_idx) < 50 or len(test_idx) < 10:
            continue

        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = model.predict_proba(X.iloc[test_idx])[:, -1]
        score = metric(y.iloc[test_idx], y_pred)

        rows.append({
            'split': split_idx,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'score': score,
        })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    # Smoke test: verify CPCV produces sensible splits
    n = 1000
    dates = pd.date_range('2014-01-01', periods=n, freq='B')
    t1 = pd.Series(dates.shift(5), index=dates)  # 5-day label horizon

    n_groups, k = 6, 2
    expected_splits = len(list(combinations(range(n_groups), k)))
    print(f"Expected splits: C({n_groups}, {k}) = {expected_splits}")

    splits = list(cpcv_split(t1, n_groups=n_groups, k_test=k, embargo_pct=0.01))
    print(f"Actual splits:   {len(splits)}\n")

    for i, (tr, te) in enumerate(splits[:5]):
        purged_count = (n - len(te)) - len(tr)
        print(f"Split {i}: train={len(tr)}, test={len(te)}, purged/embargoed={purged_count}")

    print(f"\n[OK] CPCV produces {len(splits)} disjoint paths through the data.")
