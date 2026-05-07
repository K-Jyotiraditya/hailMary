"""
Options flow signal extractor.

Options traders are the most informed participants in the market.
Unusual call volume, low put/call ratio, and IV skew changes are
leading indicators that precede equity price moves.

Signals computed from yfinance options chain (current data only):
  put_call_ratio   — vol(puts) / vol(calls); < 0.7 bullish, > 1.3 bearish
  iv_skew          — avg put IV - avg call IV; high = fear premium
  unusual_call_vol — current call vol / 30d avg call vol (proxy via open interest)
  composite_score  — [-1, +1], positive = bullish options flow
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def get_options_signal(ticker: str) -> dict:
    """
    Compute options flow signal for a single ticker.
    Returns dict with put_call_ratio, iv_skew, composite_score.
    Returns neutral defaults on failure.
    """
    neutral = {
        "put_call_ratio": 1.0,
        "iv_skew": 0.0,
        "unusual_call_vol": 1.0,
        "options_composite": 0.0,
    }

    try:
        t = yf.Ticker(ticker)
        exp_dates = t.options
        if not exp_dates:
            return neutral

        # Use nearest 2 expiries for more volume
        chains = []
        for exp in exp_dates[:2]:
            try:
                chains.append(t.option_chain(exp))
            except Exception:
                continue

        if not chains:
            return neutral

        calls = pd.concat([c.calls for c in chains], ignore_index=True)
        puts  = pd.concat([c.puts  for c in chains], ignore_index=True)

        # Filter to liquid contracts (open interest > 10)
        calls = calls[calls["openInterest"] > 10].copy()
        puts  = puts [puts ["openInterest"] > 10].copy()

        if calls.empty or puts.empty:
            return neutral

        # ── Put/call ratio ──────────────────────────────────────────────────
        call_vol = calls["volume"].fillna(0).sum()
        put_vol  = puts ["volume"].fillna(0).sum()
        pc_ratio = float(put_vol / call_vol) if call_vol > 0 else 1.0

        # ── IV skew (put IV premium over call IV) ───────────────────────────
        call_iv = calls["impliedVolatility"].replace(0, np.nan).median()
        put_iv  = puts ["impliedVolatility"].replace(0, np.nan).median()
        iv_skew = float(put_iv - call_iv) if (pd.notna(put_iv) and pd.notna(call_iv)) else 0.0

        # ── Unusual call volume ─────────────────────────────────────────────
        # Proxy: call volume vs call open interest ratio
        # High ratio = fresh demand vs. existing positions
        call_oi = calls["openInterest"].sum()
        unusual_call = float(call_vol / call_oi) if call_oi > 0 else 1.0
        unusual_call = min(unusual_call, 5.0)  # cap at 5x

        # ── Composite score [-1, +1] ────────────────────────────────────────
        # Low P/C → bullish, Low IV skew → bullish, High unusual call vol → bullish
        pc_score       = np.clip((1.0 - pc_ratio) / 0.6, -1, 1)   # neutral at 1.0
        skew_score     = np.clip(-iv_skew / 0.10, -1, 1)           # neutral at 0
        unusual_score  = np.clip((unusual_call - 1.0) / 1.5, -1, 1)

        composite = float(0.4 * pc_score + 0.4 * skew_score + 0.2 * unusual_score)

        return {
            "put_call_ratio":    round(pc_ratio, 3),
            "iv_skew":           round(iv_skew, 4),
            "unusual_call_vol":  round(unusual_call, 3),
            "options_composite": round(composite, 3),
        }

    except Exception as e:
        print(f"[Options] {ticker}: {e}")
        return neutral
