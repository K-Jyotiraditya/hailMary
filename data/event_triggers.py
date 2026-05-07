"""
Event-Driven Allocation Triggers.

Rather than trying to predict every stock every day (where edge is thin),
only deploy capital when a specific high-probability event fires:

  PEAD  — Post-Earnings Announcement Drift
          After a >5% EPS beat, stocks show documented 2-3% average drift
          over the next 20-30 days. Pure rules, no ML required.
          Source: Bernard & Thomas (1989), Da et al. (2011).

  INSIDER_CLUSTER — Multiple insiders buying in same 30-day window
          Corporate insiders don't buy unless they expect gains.
          2+ insiders buying = statistically significant cluster.
          Source: Seyhun (1998), Lakonishok & Lee (2001).

These are standalone signals that bypass the ML model entirely.
They produce force-buy recommendations with fixed allocations.
"""
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

PEAD_BEAT_THRESHOLD = 0.05    # 5% EPS beat required
PEAD_LOOKBACK_DAYS  = 45      # only trigger if earnings were within 45 days
PEAD_ALLOCATION     = 0.06    # 6% of portfolio per PEAD trigger
PEAD_HOLD_DAYS      = 25      # expected hold period

INSIDER_MIN_BUYERS  = 2       # 2+ distinct insiders required
INSIDER_LOOKBACK    = 45      # days to look back for cluster
INSIDER_ALLOCATION  = 0.05    # 5% of portfolio per insider trigger


def check_pead_trigger(ticker: str) -> dict:
    """
    Returns PEAD trigger if recent earnings beat >5% within PEAD_LOOKBACK_DAYS.
    Dict keys: triggered (bool), surprise_pct, days_since_earnings, allocation, reason
    """
    null = {"triggered": False, "surprise_pct": 0.0, "days_since_earnings": None,
            "allocation": 0.0, "reason": "no_recent_beat"}
    try:
        import yfinance as yf
        tkr = yf.Ticker(ticker)
        hist = tkr.earnings_history

        if hist is None or hist.empty:
            return null

        hist.index = pd.to_datetime(hist.index, errors="coerce")
        hist = hist.sort_index(ascending=False)

        # Look for recent beat
        for dt, row in hist.iterrows():
            if pd.isna(dt):
                continue
            days_ago = (datetime.now() - dt.to_pydatetime().replace(tzinfo=None)).days
            if days_ago > PEAD_LOOKBACK_DAYS:
                break  # sorted descending, earlier entries are older

            # Compute EPS surprise
            actual   = row.get("epsActual",   row.get("Reported EPS",  None))
            estimate = row.get("epsEstimate", row.get("EPS Estimate",  None))

            if actual is None or estimate is None:
                continue
            try:
                actual, estimate = float(actual), float(estimate)
            except (TypeError, ValueError):
                continue

            if estimate == 0 or np.isnan(actual) or np.isnan(estimate):
                continue

            surprise_pct = (actual - estimate) / abs(estimate)

            if surprise_pct >= PEAD_BEAT_THRESHOLD:
                return {
                    "triggered":          True,
                    "surprise_pct":       round(surprise_pct, 4),
                    "days_since_earnings": days_ago,
                    "allocation":         PEAD_ALLOCATION,
                    "reason":             (f"EPS beat {surprise_pct:.1%} "
                                           f"({days_ago}d ago) — PEAD drift expected"),
                }
        return null

    except Exception as e:
        print(f"  [PEAD trigger] {ticker}: {e}")
        return null


def check_insider_cluster(ticker: str) -> dict:
    """
    Returns insider cluster trigger if 2+ distinct insiders bought within INSIDER_LOOKBACK days.
    Dict keys: triggered (bool), n_insiders, total_value, allocation, reason
    """
    null = {"triggered": False, "n_insiders": 0, "total_value": 0.0,
            "allocation": 0.0, "reason": "no_insider_cluster"}
    try:
        import yfinance as yf
        tkr = yf.Ticker(ticker)
        df = tkr.insider_transactions

        if df is None or df.empty:
            return null

        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        date_col    = next((c for c in df.columns if "date" in c), None)
        text_col    = next((c for c in df.columns if "transaction" in c or "text" in c), None)
        insider_col = next((c for c in df.columns if "insider" in c or "name" in c), None)
        value_col   = next((c for c in df.columns if "value" in c), None)

        if not date_col or not text_col:
            return null

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        cutoff = datetime.now() - timedelta(days=INSIDER_LOOKBACK)
        recent = df[df[date_col] >= cutoff]

        buy_mask = recent[text_col].str.lower().str.contains(
            "purchase|buy|acquisition", na=False
        )
        buys = recent[buy_mask]

        if buys.empty:
            return null

        n_insiders = buys[insider_col].nunique() if insider_col else len(buys)

        total_value = 0.0
        if value_col and value_col in buys.columns:
            vals = pd.to_numeric(buys[value_col], errors="coerce").abs()
            total_value = float(vals.sum())

        if n_insiders >= INSIDER_MIN_BUYERS:
            return {
                "triggered":   True,
                "n_insiders":  int(n_insiders),
                "total_value": round(total_value, 0),
                "allocation":  INSIDER_ALLOCATION,
                "reason":      (f"{n_insiders} insiders bought "
                                f"(${total_value:,.0f} total) — cluster signal"),
            }
        return null

    except Exception as e:
        print(f"  [Insider trigger] {ticker}: {e}")
        return null


def get_all_event_signals(tickers: list, max_event_weight: float = 0.35) -> dict:
    """
    Check all tickers for PEAD and insider cluster signals.

    Returns dict:
      {
        ticker: {
          "events": [...],          # list of triggered event dicts
          "total_allocation": float, # sum of recommended allocations
          "reasons": [str, ...]      # human-readable explanations
        }
      }
    Only tickers with at least one trigger are included.
    Total event allocation is capped at max_event_weight of portfolio.
    """
    triggered = {}
    total_allocated = 0.0

    for ticker in tickers:
        events = []

        pead = check_pead_trigger(ticker)
        if pead["triggered"]:
            events.append(pead)

        insider = check_insider_cluster(ticker)
        if insider["triggered"]:
            events.append(insider)

        if events:
            alloc = sum(e["allocation"] for e in events)
            triggered[ticker] = {
                "events":           events,
                "total_allocation": round(alloc, 4),
                "reasons":          [e["reason"] for e in events],
            }
            total_allocated += alloc

    # Cap total event allocation
    if total_allocated > max_event_weight and triggered:
        scale = max_event_weight / total_allocated
        for t in triggered:
            triggered[t]["total_allocation"] = round(
                triggered[t]["total_allocation"] * scale, 4
            )

    return triggered


def summarize_events(event_signals: dict) -> str:
    if not event_signals:
        return "  No event-driven signals today."
    lines = [f"  Event-driven signals ({len(event_signals)} stocks):"]
    for ticker, info in event_signals.items():
        alloc = info["total_allocation"]
        reasons = " | ".join(info["reasons"])
        lines.append(f"    {ticker}: {alloc:.1%} — {reasons}")
    return "\n".join(lines)
