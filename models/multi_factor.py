"""
Multi-factor scoring: momentum, quality, value, low-volatility.
Pure math — no LLM, no training. Returns a composite health score 0-100.
Used by fundamentals_agent.py as a drop-in replacement for Ollama inference.
"""
import sys
import numpy as np
import yfinance as yf
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def _sigmoid(x: float, center: float, scale: float) -> float:
    if not np.isfinite(x):
        return 50.0
    return float(100 / (1 + np.exp(-((x - center) / scale))))


def compute_factor_score(ticker: str) -> dict:
    """
    Return a factor-based health dict compatible with FundamentalsAgent output:
      health_score (0-100), key_strength, key_risk,
      plus sub-scores: momentum_score, quality_score, value_score, vol_score.
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1y", auto_adjust=True)
        info = t.info or {}
    except Exception as e:
        return _fallback(f"data fetch failed: {e}")

    if hist.empty or len(hist) < 60:
        return _fallback("insufficient price history")

    close = hist["Close"].dropna()

    # ── Momentum factor (12-1 month) ──────────────────────────────────────────
    ret_252 = close.pct_change(min(252, len(close) - 1)).iloc[-1]
    ret_21  = close.pct_change(21).iloc[-1]
    mom_12_1 = float(ret_252 - ret_21) if np.isfinite(ret_252) and np.isfinite(ret_21) else 0.0
    momentum_score = _sigmoid(mom_12_1, center=0.05, scale=0.25)

    # ── Low-volatility factor ─────────────────────────────────────────────────
    ann_vol = float(close.pct_change().rolling(min(252, len(close))).std().iloc[-1] * np.sqrt(252))
    vol_score = _sigmoid(-ann_vol, center=-0.30, scale=0.15)

    # ── Quality factor (ROE, profit margin, revenue growth) ───────────────────
    q_inputs = []
    roe = info.get("returnOnEquity")
    if roe is not None and np.isfinite(roe):
        q_inputs.append(_sigmoid(roe, center=0.12, scale=0.15))
    pm = info.get("profitMargins")
    if pm is not None and np.isfinite(pm):
        q_inputs.append(_sigmoid(pm, center=0.10, scale=0.12))
    rg = info.get("revenueGrowth")
    if rg is not None and np.isfinite(rg):
        q_inputs.append(_sigmoid(rg, center=0.05, scale=0.10))
    quality_score = float(np.mean(q_inputs)) if q_inputs else 50.0

    # ── Value factor (P/E, P/B) ───────────────────────────────────────────────
    v_inputs = []
    pe = info.get("trailingPE")
    if pe is not None and np.isfinite(pe) and 0 < pe < 500:
        v_inputs.append(_sigmoid(-pe, center=-20, scale=15))
    pb = info.get("priceToBook")
    if pb is not None and np.isfinite(pb) and pb > 0:
        v_inputs.append(_sigmoid(-pb, center=-3, scale=3))
    value_score = float(np.mean(v_inputs)) if v_inputs else 50.0

    # ── Composite (weights from academic factor research) ────────────────────
    health = (
        0.30 * momentum_score +
        0.35 * quality_score +
        0.20 * value_score +
        0.15 * vol_score
    )
    health = float(np.clip(health, 0, 100))

    # ── Narrative ────────────────────────────────────────────────────────────
    strength = ""
    risk = ""

    if mom_12_1 > 0.20:
        strength = f"strong 12-1m momentum (+{mom_12_1:.0%})"
    elif quality_score > 65:
        strength = "high-quality fundamentals (ROE/margins)"
    elif value_score > 65:
        strength = "attractive valuation"

    if ann_vol > 0.45:
        risk = f"high volatility ({ann_vol:.0%} annualised)"
    elif pe is not None and pe > 45:
        risk = f"elevated P/E ({pe:.1f}x)"
    elif mom_12_1 < -0.10:
        risk = f"negative price momentum ({mom_12_1:.0%})"

    return {
        "health_score":    round(health),
        "momentum_score":  round(momentum_score),
        "quality_score":   round(quality_score),
        "value_score":     round(value_score),
        "vol_score":       round(vol_score),
        "key_strength":    strength,
        "key_risk":        risk,
    }


def _fallback(reason: str) -> dict:
    return {
        "health_score": 50,
        "momentum_score": 50,
        "quality_score": 50,
        "value_score": 50,
        "vol_score": 50,
        "key_strength": "",
        "key_risk": reason,
    }
