"""
Macro Regime Detector.

Combines three signals into a regime label and equity allocation multiplier:

  1. VIX level         — fear gauge (>25 = caution, >35 = risk-off)
  2. Yield curve       — 10Y minus 2Y spread via ^TNX / ^IRX
                         (inverted = warning, deeply inverted = risk-off)
  3. Credit stress     — HYG (high yield) vs LQD (investment grade) return spread
                         (HY underperforming = risk-off)

Regime outputs:
  risk_on      : equity_mult = 1.0  (full exposure)
  neutral      : equity_mult = 0.7  (reduced exposure)
  risk_off     : equity_mult = 0.4  (minimal exposure, mostly cash)

Cached for 1 day since macro conditions change slowly.
"""
import sys
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

CACHE_PATH = Path("data/macro_regime_cache.json")
CACHE_TTL_HOURS = 6

VIX_CAUTION = 20.0     # above this: neutral
VIX_RISK_OFF = 28.0    # above this: risk_off

YIELD_CURVE_CAUTION = 0.0    # below this (flat/inverted): neutral
YIELD_CURVE_RISK_OFF = -0.5  # below this (deeply inverted): risk-off

CREDIT_CAUTION = -0.02    # HYG underperforming LQD by >2%: neutral
CREDIT_RISK_OFF = -0.04   # by >4%: risk-off

EQUITY_MULT = {
    "risk_on":  1.0,
    "neutral":  0.7,
    "risk_off": 0.4,
}


def _load_regime_cache() -> dict | None:
    if not CACHE_PATH.exists():
        return None
    try:
        with open(CACHE_PATH) as f:
            data = json.load(f)
        cached_time = datetime.fromisoformat(data["timestamp"])
        if datetime.now() - cached_time < timedelta(hours=CACHE_TTL_HOURS):
            return data
    except Exception:
        pass
    return None


def _save_regime_cache(result: dict):
    CACHE_PATH.parent.mkdir(exist_ok=True)
    result["timestamp"] = datetime.now().isoformat()
    with open(CACHE_PATH, "w") as f:
        json.dump(result, f, indent=2)


def get_macro_regime(force_refresh: bool = False) -> dict:
    """
    Compute current macro regime.

    Returns dict:
      regime           : str   — "risk_on" | "neutral" | "risk_off"
      equity_mult      : float — multiplier for equity allocation (0.4-1.0)
      vix              : float — current VIX level
      yield_spread     : float — 10Y-2Y spread in pct
      credit_spread    : float — HYG vs LQD 20d return spread
      signals          : list  — individual signal labels for transparency
    """
    if not force_refresh:
        cached = _load_regime_cache()
        if cached:
            return cached

    default = {
        "regime": "neutral",
        "equity_mult": 0.7,
        "vix": 20.0,
        "yield_spread": 0.0,
        "credit_spread": 0.0,
        "signals": ["default_neutral"],
    }

    try:
        tickers = ["^VIX", "^TNX", "^IRX", "HYG", "LQD"]
        raw = yf.download(tickers, period="3mo", auto_adjust=True,
                          progress=False, threads=False)

        if not isinstance(raw.columns, pd.MultiIndex) or raw.empty:
            return default

        close = raw["Close"]

        # 1. VIX
        vix = float(close["^VIX"].dropna().iloc[-1]) if "^VIX" in close.columns else 20.0

        # 2. Yield curve: 10Y minus 2Y (in %)
        tnx = float(close["^TNX"].dropna().iloc[-1]) if "^TNX" in close.columns else 4.0
        irx = float(close["^IRX"].dropna().iloc[-1]) if "^IRX" in close.columns else 4.5
        # ^IRX is the 13-week T-bill, used as proxy for 2Y
        # ^TNX is the 10Y yield — both already in percent
        yield_spread = tnx - irx  # positive = normal, negative = inverted

        # 3. Credit spread: HYG vs LQD 20-day return
        credit_spread = 0.0
        if "HYG" in close.columns and "LQD" in close.columns:
            hyg = close["HYG"].dropna()
            lqd = close["LQD"].dropna()
            if len(hyg) >= 21 and len(lqd) >= 21:
                hyg_ret = float(hyg.iloc[-1] / hyg.iloc[-21] - 1)
                lqd_ret = float(lqd.iloc[-1] / lqd.iloc[-21] - 1)
                credit_spread = hyg_ret - lqd_ret  # negative = HY underperforming

        # Score each signal: 0 = risk_on, 1 = caution, 2 = risk_off
        scores = []
        signal_labels = []

        # VIX signal
        if vix > VIX_RISK_OFF:
            scores.append(2)
            signal_labels.append(f"vix_risk_off({vix:.1f})")
        elif vix > VIX_CAUTION:
            scores.append(1)
            signal_labels.append(f"vix_caution({vix:.1f})")
        else:
            scores.append(0)
            signal_labels.append(f"vix_ok({vix:.1f})")

        # Yield curve signal
        if yield_spread < YIELD_CURVE_RISK_OFF:
            scores.append(2)
            signal_labels.append(f"yield_risk_off({yield_spread:.2f})")
        elif yield_spread < YIELD_CURVE_CAUTION:
            scores.append(1)
            signal_labels.append(f"yield_caution({yield_spread:.2f})")
        else:
            scores.append(0)
            signal_labels.append(f"yield_ok({yield_spread:.2f})")

        # Credit signal
        if credit_spread < CREDIT_RISK_OFF:
            scores.append(2)
            signal_labels.append(f"credit_risk_off({credit_spread:.3f})")
        elif credit_spread < CREDIT_CAUTION:
            scores.append(1)
            signal_labels.append(f"credit_caution({credit_spread:.3f})")
        else:
            scores.append(0)
            signal_labels.append(f"credit_ok({credit_spread:.3f})")

        # Aggregate: any risk_off triggers risk_off; majority caution = neutral
        if max(scores) >= 2:
            regime = "risk_off"
        elif sum(s >= 1 for s in scores) >= 2:
            regime = "neutral"
        else:
            regime = "risk_on"

        result = {
            "regime":        regime,
            "equity_mult":   EQUITY_MULT[regime],
            "vix":           round(vix, 2),
            "yield_spread":  round(yield_spread, 4),
            "credit_spread": round(credit_spread, 4),
            "signals":       signal_labels,
        }
        _save_regime_cache(result)
        return result

    except Exception as e:
        print(f"  [Macro] Regime detection failed: {e}")
        return default
