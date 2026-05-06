"""
Feature Selection — keep only features with STABLE positive IC across years.

Why this matters more than gain-importance:
  Gain importance tells you which features the tree splits on. It says nothing
  about whether splitting on that feature predicts FUTURE returns. A noisy
  feature with high in-sample importance can degrade out-of-sample accuracy.
  IC stability — measured per year on held-out data — is the right criterion
  for inclusion: features that have a consistent (positive or negative)
  relationship with forward returns ACROSS multiple years.

Method:
  1. For each candidate feature, compute its yearly OOS IC against the rank
     label across all training years.
  2. Score features by:
       - Mean |IC| across years (signal strength)
       - Hit rate (% of years with same-sign IC)  (stability)
       - Sharpe-like metric: mean / std of yearly IC (information ratio)
  3. Drop features with low hit-rate OR very low mean |IC|.
"""
import numpy as np
import pandas as pd


def feature_yearly_ic(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Compute IC of each feature against the rank label, per year.

    Args:
        X: Panel features, MultiIndex (date, symbol).
        y: Rank label, MultiIndex (date, symbol).

    Returns:
        DataFrame indexed by year, columns = features, values = year IC.
    """
    dates = X.index.get_level_values('date')
    years = dates.year
    unique_years = sorted(set(years))

    rows = {}
    for year in unique_years:
        mask = (years == year)
        if mask.sum() < 100:
            continue
        X_y = X[mask]
        y_y = y[mask]

        # Per-date Spearman correlation, then average
        df = X_y.copy()
        df['__y__'] = y_y.values
        ics = {}
        for col in X.columns:
            valid = df[[col, '__y__']].dropna()
            if len(valid) < 50:
                ics[col] = np.nan
                continue
            try:
                ics[col] = valid[col].corr(valid['__y__'], method='spearman')
            except Exception:
                ics[col] = np.nan
        rows[year] = ics

    return pd.DataFrame(rows).T


def score_features(yearly_ic: pd.DataFrame) -> pd.DataFrame:
    """
    Score each feature by stability + magnitude of IC.

    Returns DataFrame with columns: mean_ic, mean_abs_ic, hit_rate, ic_ir.
    """
    n_years = len(yearly_ic)
    scores = pd.DataFrame({
        'mean_ic':     yearly_ic.mean(),
        'mean_abs_ic': yearly_ic.abs().mean(),
        'std_ic':      yearly_ic.std(),
        'min_ic':      yearly_ic.min(),
        'max_ic':      yearly_ic.max(),
    })

    # Same-sign hit rate: fraction of years where the IC has the same sign as the mean
    sign_of_mean = np.sign(scores['mean_ic']).replace(0, 1)
    aligned = yearly_ic.apply(lambda col: np.sign(col).replace(0, sign_of_mean[col.name]) == sign_of_mean[col.name])
    scores['hit_rate'] = aligned.mean()

    # IC Information Ratio: mean / std of yearly IC
    scores['ic_ir'] = scores['mean_ic'] / scores['std_ic'].replace(0, np.nan)

    return scores.sort_values('mean_abs_ic', ascending=False)


def select_features(scores: pd.DataFrame,
                    min_mean_abs_ic: float = 0.005,
                    min_hit_rate: float = 0.55,
                    top_n: int = None) -> list:
    """
    Pick features that pass IC criteria.

    Args:
        min_mean_abs_ic: Require |IC| above this threshold.
        min_hit_rate: Require same-sign IC in >= this fraction of years.
        top_n: If set, also cap total features at this number (by mean_abs_ic).

    Returns:
        List of selected feature names.
    """
    keep = scores[
        (scores['mean_abs_ic'] >= min_mean_abs_ic) &
        (scores['hit_rate'] >= min_hit_rate)
    ].index.tolist()

    if top_n is not None and len(keep) > top_n:
        # Re-rank by mean_abs_ic among the qualifying ones
        keep = scores.loc[keep].sort_values('mean_abs_ic', ascending=False).head(top_n).index.tolist()

    return keep


def select_features_walk_forward(X: pd.DataFrame, y: pd.Series,
                                 train_through: pd.Timestamp,
                                 min_mean_abs_ic: float = 0.005,
                                 min_hit_rate: float = 0.55,
                                 top_n: int = 80) -> list:
    """
    Compute IC stability using only data BEFORE train_through, then select.
    Used for a single walk-forward year so we don't peek at the future.
    """
    dates = X.index.get_level_values('date')
    train_mask = dates < train_through
    if train_mask.sum() < 5000:
        return X.columns.tolist()  # Not enough data, keep all

    yearly = feature_yearly_ic(X[train_mask], y[train_mask])
    if yearly.empty:
        return X.columns.tolist()

    scores = score_features(yearly)
    return select_features(scores, min_mean_abs_ic=min_mean_abs_ic,
                           min_hit_rate=min_hit_rate, top_n=top_n)


if __name__ == '__main__':
    # Run a smoke test
    cs_panel = pd.read_parquet('data/processed/features/CS_PANEL.parquet')
    market_context = pd.read_parquet('data/processed/features/MARKET_CONTEXT.parquet')
    from ml.cs_model import assemble_panel_features, align_panel_features_labels
    from labels.rank_label import build_label_panel
    panel = assemble_panel_features(cs_panel, market_context)
    labels = build_label_panel(horizon=5, method='pct_rank')
    X, y = align_panel_features_labels(panel, labels, feature_lag=1)

    print(f"Panel: {X.shape}")
    print("Computing yearly IC for each of 144 features (this takes ~30s)...")
    yearly = feature_yearly_ic(X, y)
    scores = score_features(yearly)

    print(f"\nTop 25 features by mean |IC|:")
    print(scores.head(25).round(4).to_string())

    selected = select_features(scores, min_mean_abs_ic=0.005, min_hit_rate=0.55)
    print(f"\nSelected {len(selected)}/{len(scores)} features by stability + magnitude")
    print(f"Dropped: {[f for f in X.columns if f not in selected][:20]}")
