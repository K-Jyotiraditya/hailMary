"""
Regime Filter — go to cash when the broad market is in a downtrend.

Phase 1's biggest weakness was the COVID/2024 drawdowns: -42% max DD because
the strategy was always 100% invested even when the broader market was
collapsing. This module gives the strategy a "circuit breaker" — when the
market regime is unfavorable, hold cash.

CLASSIC RULE (Faber 2007 / many others): NIFTY 50 close vs its 200-day SMA.
  - Above 200-DMA: bullish regime, deploy strategy.
  - Below 200-DMA: bearish regime, hold cash.

This is the simplest, most-tested regime filter in tactical asset allocation.
It avoids the worst drawdowns at the cost of being slow to re-enter rallies.
"""
import pandas as pd


def sma_regime(nifty_close: pd.Series, window: int = 200,
               require_strict: bool = False) -> pd.Series:
    """
    Boolean Series indicating whether the regime is BULLISH on each date.

    Args:
        nifty_close: NIFTY 50 close price series.
        window: SMA window (200 = traditional, 50 = more reactive).
        require_strict: If True, also require positive 20d return (extra filter).
    """
    sma = nifty_close.rolling(window).mean()
    bullish = nifty_close > sma
    if require_strict:
        bullish = bullish & (nifty_close.pct_change(20) > 0)
    return bullish.fillna(False)


def is_bullish(nifty_close: pd.Series, date: pd.Timestamp,
               window: int = 200) -> bool:
    """Snapshot regime check for a specific date."""
    prior = nifty_close.loc[:date]
    if len(prior) < window:
        return True  # Default to bullish if insufficient history
    sma = prior.tail(window).mean()
    return bool(prior.iloc[-1] > sma)


def apply_regime_filter(weights: pd.Series, is_bull: bool,
                        bear_exposure: float = 0.0) -> pd.Series:
    """
    Apply the regime filter to a weight vector.

    Args:
        weights: Strategy's recommended weights (sum <= 1).
        is_bull: True if regime is bullish.
        bear_exposure: How much of original weights to retain in bear regime.
                       Default 0 = full cash. Could use 0.3 = "hedge" position.
    """
    if is_bull:
        return weights
    return weights * bear_exposure


if __name__ == '__main__':
    nifty = pd.read_parquet('data/processed/BENCHMARK_INDEX.parquet')['Close']
    regime = sma_regime(nifty, window=200)

    print(f"NIFTY 50 regime (200-DMA filter):")
    print(f"  Total trading days: {len(regime)}")
    print(f"  Bullish days:       {regime.sum()} ({regime.mean():.1%})")
    print(f"  Bearish days:       {(~regime).sum()} ({(~regime).mean():.1%})")

    # Show transitions
    transitions = regime.astype(int).diff().abs() > 0
    print(f"  Regime transitions: {transitions.sum()}")

    # Show last 10 transitions
    trans_dates = regime.index[transitions]
    print(f"\nLast 10 regime transitions:")
    for d in trans_dates[-10:]:
        new_state = "BULL" if regime.loc[d] else "BEAR"
        print(f"  {d.date()} -> {new_state}")
