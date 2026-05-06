"""
Market-Context Features — global market state at each date.

These tell the model what the BROADER MARKET is doing. The same per-stock
RSI=70 has different implications in a bull vs bear market; market-context
features give the model that info.

Features computed from NIFTY 50 index OHLCV:
  - Trend regime: NIFTY close vs SMA200 (binary + magnitude)
  - Realized vol: 20d, 60d annualized
  - Vol regime: 20d / 60d ratio (rising vs falling vol)
  - Breadth: fraction of universe above their own SMA200
  - Drawdown: distance from 252d high
  - Momentum: 20d, 60d, 252d returns
"""
import pandas as pd
import numpy as np
from pathlib import Path


def market_features(nifty_df: pd.DataFrame) -> pd.DataFrame:
    """Compute market-wide features from the NIFTY 50 index OHLCV."""
    out = pd.DataFrame(index=nifty_df.index)
    close = nifty_df['Close']
    daily_ret = close.pct_change()

    # Trend regime
    sma_200 = close.rolling(200).mean()
    sma_50 = close.rolling(50).mean()
    out['mkt_above_sma200'] = (close > sma_200).astype(int)
    out['mkt_above_sma50'] = (close > sma_50).astype(int)
    out['mkt_close_vs_sma200'] = close / sma_200 - 1
    out['mkt_close_vs_sma50'] = close / sma_50 - 1
    out['mkt_sma50_vs_sma200'] = sma_50 / sma_200 - 1  # Golden / death cross strength

    # Volatility regime
    out['mkt_vol_20'] = daily_ret.rolling(20).std() * np.sqrt(252)
    out['mkt_vol_60'] = daily_ret.rolling(60).std() * np.sqrt(252)
    out['mkt_vol_ratio'] = out['mkt_vol_20'] / out['mkt_vol_60']

    # Drawdown from 252d peak
    rolling_peak = close.rolling(252, min_periods=1).max()
    out['mkt_drawdown_252'] = (close - rolling_peak) / rolling_peak

    # Momentum
    out['mkt_ret_20d'] = close.pct_change(20)
    out['mkt_ret_60d'] = close.pct_change(60)
    out['mkt_ret_252d'] = close.pct_change(252)

    return out


def breadth_features(prices_dict: dict, lookback: int = 200) -> pd.Series:
    """
    Market breadth: fraction of universe trading above their own SMA(lookback).

    Strong bull markets have high breadth (most stocks rallying). Narrow
    rallies (low breadth) often precede pullbacks.
    """
    above_ma = pd.DataFrame()
    for symbol, prices in prices_dict.items():
        sma = prices.rolling(lookback).mean()
        above_ma[symbol] = (prices > sma).astype(int)

    breadth = above_ma.mean(axis=1)
    breadth.name = f'mkt_breadth_{lookback}'
    return breadth


def build_market_context(
    nifty_path: str = 'data/processed/BENCHMARK_INDEX.parquet',
    prices_dir: str = 'data/processed',
    save_path: str = 'data/processed/features/MARKET_CONTEXT.parquet',
) -> pd.DataFrame:
    """
    Build the daily market-context DataFrame indexed by date.

    Returns DataFrame with one row per trading date, columns prefixed `mkt_`.
    """
    nifty = pd.read_parquet(nifty_path)
    out = market_features(nifty)

    # Breadth across our universe
    prices_dict = {}
    for fp in sorted(Path(prices_dir).glob('*.parquet')):
        if fp.stem == 'BENCHMARK_INDEX':
            continue
        prices_dict[fp.stem] = pd.read_parquet(fp)['Close']

    out['mkt_breadth_50'] = breadth_features(prices_dict, lookback=50)
    out['mkt_breadth_200'] = breadth_features(prices_dict, lookback=200)
    out = out.dropna()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(save_path)
        print(f"Saved market context to {save_path}: {out.shape}")

    return out


if __name__ == '__main__':
    mc = build_market_context()
    print(f"\nMarket context features: {mc.columns.tolist()}")
    print(f"\nLast 3 rows:")
    print(mc.tail(3).round(4).to_string())
