"""
Cross-sectional Features — rank each per-stock feature across the universe per date.

WHY:
  Per-stock feature values like RSI=70 mean different things depending on the
  state of the broader market. RSI=70 in a bull market (where the average stock
  has RSI=65) signals less than RSI=70 when the average is 50. Cross-sectional
  ranks normalize each feature INTO THE PRESENT MARKET — a more comparable
  signal across regimes.

  These are the bread-and-butter of equity ranking: pick stocks that are
  RELATIVELY strong, not absolutely strong. Phase 1's IC ≈ 0.02 was on absolute
  features. Cross-sectional ranks usually do meaningfully better.

OUTPUT:
  A panel DataFrame with MultiIndex (date, symbol) where each column is a
  feature ranked into [0, 1] within its date.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def stack_to_panel(per_stock_features: dict) -> pd.DataFrame:
    """
    Combine per-stock feature DataFrames into a single panel with MultiIndex
    (date, symbol). All inputs must have the SAME columns.
    """
    parts = []
    for symbol, df in per_stock_features.items():
        d = df.copy()
        d.index = pd.MultiIndex.from_product(
            [d.index, [symbol]], names=['date', 'symbol']
        )
        parts.append(d)
    panel = pd.concat(parts, axis=0).sort_index()
    return panel


def cross_sectional_rank(panel: pd.DataFrame, prefix: str = 'cs_') -> pd.DataFrame:
    """
    For each (date, feature), rank stocks into [0, 1].

    Args:
        panel: DataFrame with MultiIndex (date, symbol).
        prefix: Prefix for the new column names (e.g., 'cs_rsi_14').

    Returns:
        DataFrame with same shape, ranked columns prefixed with `prefix`.
    """
    # Per-date ranking: rank within each date level, normalize to [0, 1]
    ranked = panel.groupby(level='date').rank(pct=True)
    ranked.columns = [f'{prefix}{c}' for c in ranked.columns]
    return ranked


def cross_sectional_zscore(panel: pd.DataFrame, prefix: str = 'cz_') -> pd.DataFrame:
    """
    Cross-sectional z-score: per date, demean and divide by cross-sectional std.
    Useful when you want sign and magnitude (not just rank).
    """
    grouped = panel.groupby(level='date')
    mean = grouped.transform('mean')
    std = grouped.transform('std')
    zs = (panel - mean) / std.replace(0, np.nan)
    zs.columns = [f'{prefix}{c}' for c in zs.columns]
    return zs


def build_cross_sectional_features(
    features_dir: str = 'data/processed/features',
    suffix: str = '_ml',
    method: str = 'rank',
    save_path: str = 'data/processed/features/CS_PANEL.parquet',
) -> pd.DataFrame:
    """
    Read per-stock features, build a panel, compute cross-sectional ranks.

    Args:
        method: 'rank' (percentile rank) or 'zscore' (cross-sectional z-score)
                or 'both' to include both transformations.

    Returns:
        Panel DataFrame with MultiIndex (date, symbol), columns = original
        + cross-sectional features.
    """
    feature_files = sorted(Path(features_dir).glob(f'*{suffix}.parquet'))
    per_stock = {}
    for fp in feature_files:
        symbol = fp.stem.replace(suffix, '')
        df = pd.read_parquet(fp)
        per_stock[symbol] = df

    print(f"Stacking {len(per_stock)} stocks into panel...")
    panel = stack_to_panel(per_stock)
    print(f"  Panel shape: {panel.shape}")

    if method == 'rank':
        cs = cross_sectional_rank(panel)
        out = pd.concat([panel, cs], axis=1)
    elif method == 'zscore':
        cs = cross_sectional_zscore(panel)
        out = pd.concat([panel, cs], axis=1)
    elif method == 'both':
        ranks = cross_sectional_rank(panel)
        zs = cross_sectional_zscore(panel)
        out = pd.concat([panel, ranks, zs], axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"  Output panel: {out.shape}")
    if save_path:
        out.to_parquet(save_path)
        print(f"  Saved to {save_path}")

    return out


if __name__ == '__main__':
    panel = build_cross_sectional_features(method='rank')
    print(f"\nSample rows (last date):")
    last_date = panel.index.get_level_values('date').max()
    print(panel.loc[last_date].iloc[:5, [0, 1, 2, 65, 66, 67]].round(3))
