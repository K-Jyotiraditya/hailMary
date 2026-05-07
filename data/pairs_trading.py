"""
Statistical Arbitrage — Pairs Trading.

Finds cointegrated pairs from the watchlist using the Engle-Granger test,
then signals long/short when the spread diverges beyond 2 standard deviations.

This strategy is market-neutral: you profit from the spread converging
regardless of whether the market goes up or down.

Usage:
    pairs = find_cointegrated_pairs(WATCHLIST)
    signals = get_pairs_signals(pairs)
"""
import sys
import itertools
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

ZSCORE_ENTRY  = 2.0    # open trade when spread deviates this many sigmas
ZSCORE_EXIT   = 0.5    # close trade when spread reverts to this
MIN_LOOKBACK  = 60     # days for spread z-score computation
COINT_PVALUE  = 0.05   # significance threshold for cointegration test


def find_cointegrated_pairs(tickers: list, lookback: str = "1y") -> list:
    """
    Test all pairs in tickers for cointegration.
    Returns list of dicts sorted by p-value:
      {ticker1, ticker2, pvalue, hedge_ratio}
    """
    from statsmodels.tsa.stattools import coint

    print(f"  [Pairs] Testing {len(list(itertools.combinations(tickers, 2)))} pairs for cointegration...")
    raw = yf.download(tickers, period=lookback, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].dropna(axis=1, how="all")
    else:
        return []

    pairs = []
    for t1, t2 in itertools.combinations(prices.columns, 2):
        s1 = prices[t1].dropna()
        s2 = prices[t2].dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < 120:
            continue
        s1, s2 = s1[common], s2[common]
        try:
            _, pvalue, _ = coint(s1, s2)
            if pvalue < COINT_PVALUE:
                # hedge ratio via OLS
                from numpy.linalg import lstsq
                X = np.column_stack([s2.values, np.ones(len(s2))])
                hedge_ratio = float(lstsq(X, s1.values, rcond=None)[0][0])
                pairs.append({
                    "ticker1": t1,
                    "ticker2": t2,
                    "pvalue":  round(pvalue, 4),
                    "hedge_ratio": round(hedge_ratio, 4),
                })
        except Exception:
            continue

    pairs.sort(key=lambda x: x["pvalue"])
    print(f"  [Pairs] Found {len(pairs)} cointegrated pairs (p < {COINT_PVALUE})")
    for p in pairs[:5]:
        print(f"    {p['ticker1']}/{p['ticker2']}  p={p['pvalue']}  hedge={p['hedge_ratio']:.3f}")
    return pairs


def get_pairs_signals(pairs: list, lookback: str = "6mo") -> list:
    """
    For each pair, compute the current spread z-score and generate a signal.
    Returns list of dicts:
      {ticker1, ticker2, zscore, signal, long_ticker, short_ticker}
    signal: 'enter_long_spread' | 'enter_short_spread' | 'exit' | 'hold'
    """
    if not pairs:
        return []

    all_tickers = list({t for p in pairs for t in [p["ticker1"], p["ticker2"]]})
    raw = yf.download(all_tickers, period=lookback, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        return []

    signals = []
    for pair in pairs:
        t1, t2 = pair["ticker1"], pair["ticker2"]
        hr = pair["hedge_ratio"]

        if t1 not in prices.columns or t2 not in prices.columns:
            continue

        s1 = prices[t1].dropna()
        s2 = prices[t2].dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < MIN_LOOKBACK:
            continue
        s1, s2 = s1[common], s2[common]

        spread = s1 - hr * s2
        spread_mean = spread.rolling(MIN_LOOKBACK).mean()
        spread_std  = spread.rolling(MIN_LOOKBACK).std()
        zscore = float((spread.iloc[-1] - spread_mean.iloc[-1]) / spread_std.iloc[-1])

        if not np.isfinite(zscore):
            continue

        if zscore > ZSCORE_ENTRY:
            # spread too high: t1 overvalued vs t2 → short t1, long t2
            signal = "enter_short_spread"
            long_t, short_t = t2, t1
        elif zscore < -ZSCORE_ENTRY:
            # spread too low: t1 undervalued vs t2 → long t1, short t2
            signal = "enter_long_spread"
            long_t, short_t = t1, t2
        elif abs(zscore) < ZSCORE_EXIT:
            signal = "exit"
            long_t, short_t = None, None
        else:
            signal = "hold"
            long_t, short_t = None, None

        signals.append({
            "ticker1":      t1,
            "ticker2":      t2,
            "zscore":       round(zscore, 3),
            "signal":       signal,
            "long_ticker":  long_t,
            "short_ticker": short_t,
            "pvalue":       pair["pvalue"],
        })

    return signals


def pairs_to_weight_adjustments(signals: list, base_allocation: float = 0.05) -> dict:
    """
    Convert pair signals into portfolio weight adjustments.
    Returns dict of {ticker: weight_delta} to add on top of existing weights.
    Only uses 'enter_long_spread' signals (no shorting in paper account by default).
    """
    adjustments = {}
    for sig in signals:
        if sig["signal"] == "enter_long_spread" and sig["long_ticker"]:
            t = sig["long_ticker"]
            adjustments[t] = adjustments.get(t, 0.0) + base_allocation
    return adjustments
