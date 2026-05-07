"""
Google Trends spike detector.

A spike in search interest for a stock ticker precedes retail buying
by 3-5 days (documented in Da, Engelberg & Gao, 2011, Journal of Finance).

spike_ratio = current_week_interest / 90d_avg_interest
spike_flag  = 1 if spike_ratio > 1.5 else 0

Uses pytrends (unofficial Google Trends API). Has rate limits:
sleep 1-2s between tickers to avoid 429s.
"""
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

# pytrends uses urllib3 < 2.0 API ('method_whitelist' renamed to 'allowed_methods').
# Patch Retry.__init__ once so pytrends works with urllib3 >= 2.0.
try:
    from urllib3.util.retry import Retry as _Retry
    _orig_retry_init = _Retry.__init__
    def _patched_retry_init(self, *args, **kwargs):
        if "method_whitelist" in kwargs:
            kwargs["allowed_methods"] = kwargs.pop("method_whitelist")
        _orig_retry_init(self, *args, **kwargs)
    _Retry.__init__ = _patched_retry_init
except Exception:
    pass


def _build_pytrends():
    from pytrends.request import TrendReq
    return TrendReq(hl="en-US", tz=300, timeout=(10, 30), retries=2, backoff_factor=0.5)


def get_trends_history(ticker: str, period_years: int = 2) -> pd.DataFrame:
    """
    Fetch weekly Google Trends interest for ticker over the past N years.
    Returns DataFrame indexed by week-start date with columns:
      interest (0-100), spike_ratio, spike_flag
    Returns empty DataFrame on failure.
    """
    try:
        pytrends = _build_pytrends()
        tf = f"today {period_years * 12}-m"
        pytrends.build_payload([ticker], cat=7, timeframe=tf)  # cat=7 = Finance
        raw = pytrends.interest_over_time()

        if raw.empty or ticker not in raw.columns:
            return pd.DataFrame()

        s = raw[ticker].astype(float)
        s.index = pd.to_datetime(s.index)

        rolling_avg = s.rolling(13, min_periods=4).mean()  # ~90 days (13 weeks)
        spike_ratio = (s / rolling_avg.replace(0, np.nan)).fillna(1.0)

        df = pd.DataFrame({
            "gtrends_interest": s,
            "gtrends_spike_ratio": spike_ratio,
            "gtrends_spike_flag": (spike_ratio > 1.5).astype(float),
        })
        return df
    except Exception as e:
        print(f"[GTrends] {ticker}: {e}")
        return pd.DataFrame()


def build_trends_features(ticker: str, date_index: pd.DatetimeIndex,
                          sleep_s: float = 1.5) -> pd.DataFrame:
    """
    Fetch weekly trends and forward-fill to daily date_index.
    Returns DataFrame with gtrends_* columns indexed by date_index.
    """
    time.sleep(sleep_s)  # respect rate limit
    weekly = get_trends_history(ticker)

    default = pd.DataFrame({
        "gtrends_interest":    1.0,
        "gtrends_spike_ratio": 1.0,
        "gtrends_spike_flag":  0.0,
    }, index=date_index)

    if weekly.empty:
        return default

    # Reindex to daily, forward-fill (each week's value holds until next week)
    daily_idx = pd.date_range(date_index.min(), date_index.max(), freq="D")
    weekly_reindexed = weekly.reindex(daily_idx).ffill().bfill()
    aligned = weekly_reindexed.reindex(date_index).ffill().bfill()

    if aligned.empty or aligned.isnull().all().all():
        return default

    return aligned.fillna({"gtrends_interest": 1.0,
                            "gtrends_spike_ratio": 1.0,
                            "gtrends_spike_flag": 0.0})


def get_live_trends(ticker: str) -> dict:
    """Current Google Trends signal for live pipeline inference."""
    time.sleep(1.0)
    weekly = get_trends_history(ticker, period_years=1)
    if weekly.empty:
        return {"gtrends_spike_ratio": 1.0, "gtrends_spike_flag": 0.0}

    latest = weekly.iloc[-1]
    return {
        "gtrends_spike_ratio": round(float(latest["gtrends_spike_ratio"]), 3),
        "gtrends_spike_flag":  float(latest["gtrends_spike_flag"]),
    }
