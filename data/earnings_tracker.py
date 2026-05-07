"""
PEAD (Post-Earnings Announcement Drift) tracker.

Fetches historical earnings surprises via yfinance and computes:
  - earnings_surprise_pct: (actual - estimate) / |estimate| * 100
  - pead_signal: decaying score [0,1] for 60 days after a >5% beat
  - days_since_beat: days elapsed since last significant beat

Used as features in XGBoost training and as a live signal in the pipeline.
PEAD is one of the most replicated anomalies in academic finance.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent.parent))

BEAT_THRESHOLD = 0.05   # 5% EPS beat triggers PEAD signal
PEAD_WINDOW    = 60     # drift window in trading days


def get_earnings_history(ticker: str) -> pd.DataFrame:
    """
    Return DataFrame indexed by earnings date with columns:
      actual, estimate, surprise_pct
    Sorted ascending by date.
    """
    try:
        t = yf.Ticker(ticker)
        eh = t.earnings_history
        if eh is None or eh.empty:
            return pd.DataFrame()

        # yfinance column names vary; normalise
        col_map = {}
        for c in eh.columns:
            cl = c.lower()
            if "reported" in cl or "actual" in cl:
                col_map[c] = "actual"
            elif "estimate" in cl or "eps estimate" in cl:
                col_map[c] = "estimate"
        eh = eh.rename(columns=col_map)

        if "actual" not in eh.columns or "estimate" not in eh.columns:
            return pd.DataFrame()

        eh = eh[["actual", "estimate"]].copy()
        eh["actual"]   = pd.to_numeric(eh["actual"],   errors="coerce")
        eh["estimate"] = pd.to_numeric(eh["estimate"], errors="coerce")
        eh = eh.dropna()

        # surprise_pct: positive = beat, negative = miss
        denom = eh["estimate"].abs().replace(0, np.nan)
        eh["surprise_pct"] = (eh["actual"] - eh["estimate"]) / denom

        eh.index = pd.to_datetime(eh.index)
        return eh.sort_index()
    except Exception as e:
        print(f"[Earnings] {ticker}: {e}")
        return pd.DataFrame()


def build_pead_features(ticker: str, date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    For each date in date_index, compute PEAD features using only
    earnings history available up to (and including) that date.
    Returns DataFrame with columns: earnings_surprise_pct, pead_signal, days_since_beat
    """
    eh = get_earnings_history(ticker)
    results = []

    for date in date_index:
        past = eh[eh.index <= date] if not eh.empty else pd.DataFrame()

        if past.empty:
            results.append({
                "date": date,
                "earnings_surprise_pct": 0.0,
                "pead_signal": 0.0,
                "days_since_beat": -1,
            })
            continue

        # Most recent earnings
        latest = past.iloc[-1]
        surprise = float(latest["surprise_pct"])
        days_since = (date - latest.name).days

        # PEAD signal: linear decay from 1→0 over PEAD_WINDOW days after a beat
        pead = 0.0
        days_since_beat = -1
        if surprise >= BEAT_THRESHOLD and 0 <= days_since <= PEAD_WINDOW:
            pead = max(0.0, 1.0 - days_since / PEAD_WINDOW) * min(1.0, surprise / 0.15)
            days_since_beat = days_since

        results.append({
            "date": date,
            "earnings_surprise_pct": round(surprise, 4),
            "pead_signal": round(pead, 4),
            "days_since_beat": days_since_beat,
        })

    return pd.DataFrame(results).set_index("date")


def get_live_pead(ticker: str) -> dict:
    """Current PEAD signal for live pipeline inference."""
    eh = get_earnings_history(ticker)
    if eh.empty:
        return {"earnings_surprise_pct": 0.0, "pead_signal": 0.0, "days_since_beat": -1}

    today = pd.Timestamp(datetime.now().date())
    latest = eh.iloc[-1]
    surprise = float(latest["surprise_pct"])
    days_since = (today - latest.name).days

    pead = 0.0
    days_since_beat = -1
    if surprise >= BEAT_THRESHOLD and 0 <= days_since <= PEAD_WINDOW:
        pead = max(0.0, 1.0 - days_since / PEAD_WINDOW) * min(1.0, surprise / 0.15)
        days_since_beat = days_since

    return {
        "earnings_surprise_pct": round(surprise, 4),
        "pead_signal": round(pead, 4),
        "days_since_beat": days_since_beat,
        "latest_earnings_date": str(latest.name.date()),
    }
