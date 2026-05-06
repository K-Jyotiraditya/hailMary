"""
Cross-sectional Rank Labels — predict RELATIVE performance.

WHY:
  Phase 1 used per-stock binary labels: "did this stock have a positive triple-
  barrier outcome?" That's an ABSOLUTE return prediction. It conflates two
  things: market beta + stock-specific alpha. When the whole market goes up,
  every stock has a positive label and there's nothing to learn.

  Cross-sectional rank labels solve this: at each date, rank all stocks in the
  universe by their forward 5-day return. Convert to percentile [0, 1]. The
  model learns "which stocks will outperform their peers next period" — that's
  pure alpha, regardless of market regime.

OUTPUT:
  Panel DataFrame indexed by (date, symbol) with the rank label as a column.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def forward_returns(close_panel: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Compute forward returns over `horizon` days for each stock.

    Args:
        close_panel: DataFrame with columns = stocks, rows = dates.
        horizon: Forward return horizon in trading days.

    Returns:
        DataFrame with same shape; values are forward returns aligned with
        the entry date (so row at date t contains the return realized over
        [t, t+horizon]).
    """
    return close_panel.pct_change(horizon).shift(-horizon)


def cross_sectional_rank_label(
    close_panel: pd.DataFrame,
    horizon: int = 5,
    method: str = 'pct_rank',
) -> pd.DataFrame:
    """
    Compute cross-sectional rank labels.

    Args:
        close_panel: DataFrame columns=stocks, rows=dates, values=Close prices.
        horizon: Forward return horizon.
        method:
          'pct_rank'    -> percentile rank in [0, 1] (regression target)
          'top_decile'  -> 1 if top 10% of stocks today, else 0 (classification)
          'top_quintile'-> 1 if top 20%, else 0

    Returns:
        DataFrame columns=stocks, rows=dates, values = label.
    """
    fwd = forward_returns(close_panel, horizon)

    # Rank within each row (each date)
    if method == 'pct_rank':
        labels = fwd.rank(axis=1, pct=True)
    elif method == 'top_decile':
        labels = fwd.rank(axis=1, pct=True)
        labels = (labels >= 0.9).astype(int)
    elif method == 'top_quintile':
        labels = fwd.rank(axis=1, pct=True)
        labels = (labels >= 0.8).astype(int)
    else:
        raise ValueError(f"Unknown method: {method}")

    return labels


def build_label_panel(
    prices_dir: str = 'data/processed',
    horizon: int = 5,
    method: str = 'pct_rank',
    exclude=('BENCHMARK_INDEX',),
) -> pd.DataFrame:
    """
    Build a full label panel from a directory of stock parquets.

    Returns:
        DataFrame with MultiIndex (date, symbol), single column 'rank_label'.
    """
    closes = {}
    for fp in sorted(Path(prices_dir).glob('*.parquet')):
        if fp.stem in exclude:
            continue
        closes[fp.stem] = pd.read_parquet(fp)['Close']

    close_panel = pd.DataFrame(closes).sort_index()
    label_panel = cross_sectional_rank_label(close_panel, horizon=horizon, method=method)

    # Convert wide -> long (panel)
    long_df = label_panel.stack().to_frame(name='rank_label')
    long_df.index.names = ['date', 'symbol']
    return long_df.dropna()


if __name__ == '__main__':
    labels = build_label_panel(horizon=5, method='pct_rank')
    print(f"Label panel: {labels.shape}")
    print(f"Date range:  {labels.index.get_level_values('date').min().date()} to "
          f"{labels.index.get_level_values('date').max().date()}")
    print(f"Symbols: {labels.index.get_level_values('symbol').nunique()}")
    print(f"\nLabel distribution:")
    print(labels['rank_label'].describe().round(4).to_string())

    # Sanity: distribution should be approximately uniform on [0, 1]
    print(f"\nMean: {labels['rank_label'].mean():.4f} (uniform expected ~0.5)")
    print(f"Std:  {labels['rank_label'].std():.4f}  (uniform expected ~0.289)")
