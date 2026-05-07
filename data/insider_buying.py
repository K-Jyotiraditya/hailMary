"""
Insider Buying Signal — Form 4 (SEC EDGAR) via yfinance.

Corporate insiders (directors, officers, >10% holders) must file Form 4
within 2 days of a transaction. Purchases by multiple insiders are a
strong positive signal — insiders rarely buy unless they expect gains.

Signal construction:
  - Fetch last 90 days of insider transactions from yfinance
  - Count purchase events and compute purchase dollar volume
  - purchase_ratio = purchases / (purchases + sales)   [0..1]
  - insider_score = weighted composite [0..1]

Note: yfinance insider data sourced from Yahoo Finance (limited depth).
For production, upgrade to SEC EDGAR XBRL API for full Form 4 history.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

LOOKBACK_DAYS = 90
MIN_PURCHASE_AMOUNT = 10_000   # ignore tiny token buys


def get_insider_signal(ticker: str) -> dict:
    """
    Fetch insider transactions for ticker and compute a composite score.

    Returns dict:
      purchase_ratio   : float [0..1] — fraction of recent transactions that were buys
      n_insiders       : int           — distinct insiders who bought
      insider_score    : float [0..1] — composite signal (0 = no signal, 1 = strong buy)
    """
    default = {"purchase_ratio": 0.0, "n_insiders": 0, "insider_score": 0.0}
    try:
        tkr = yf.Ticker(ticker)
        df = tkr.insider_transactions

        if df is None or df.empty:
            return default

        # Normalise column names (yfinance can return varying capitalisation)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Filter to last LOOKBACK_DAYS
        date_col = next((c for c in df.columns if "date" in c), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            cutoff = datetime.now() - timedelta(days=LOOKBACK_DAYS)
            df = df[df[date_col] >= cutoff]

        if df.empty:
            return default

        # Identify buy/sell transactions
        text_col = next((c for c in df.columns if "transaction" in c or "text" in c), None)
        shares_col = next((c for c in df.columns if "shares" in c), None)
        value_col = next((c for c in df.columns if "value" in c), None)
        insider_col = next((c for c in df.columns if "insider" in c or "name" in c), None)

        if text_col is None:
            return default

        buy_mask = df[text_col].str.lower().str.contains("purchase|buy|acquisition", na=False)
        sell_mask = df[text_col].str.lower().str.contains("sale|sell|disposal", na=False)

        buys = df[buy_mask].copy()
        sells = df[sell_mask].copy()

        # Filter by minimum dollar value if available
        if value_col and value_col in buys.columns:
            buys[value_col] = pd.to_numeric(buys[value_col], errors="coerce").abs()
            buys = buys[buys[value_col] >= MIN_PURCHASE_AMOUNT]

        n_buy_txns = len(buys)
        n_sell_txns = len(sells)
        total_txns = n_buy_txns + n_sell_txns

        if total_txns == 0:
            return default

        purchase_ratio = n_buy_txns / total_txns

        # Count distinct buying insiders
        n_insiders = 0
        if insider_col and insider_col in buys.columns:
            n_insiders = buys[insider_col].nunique()
        else:
            n_insiders = n_buy_txns  # fallback: each txn = 1 insider

        # Composite score
        # - purchase_ratio contribution: sigmoid-like scaling
        ratio_score = purchase_ratio ** 2          # quadratic — reward when most txns are buys
        # - breadth contribution: multiple insiders buying = stronger signal (cap at 5)
        breadth_score = min(n_insiders / 5.0, 1.0)
        insider_score = float(0.6 * ratio_score + 0.4 * breadth_score)

        return {
            "purchase_ratio": round(purchase_ratio, 3),
            "n_insiders":     int(n_insiders),
            "insider_score":  round(insider_score, 3),
        }

    except Exception as e:
        print(f"  [Insider] {ticker}: {e}")
        return default


def build_insider_features(ticker: str, date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Return a DataFrame aligned to date_index with insider_score column.
    Current signal broadcast as constant across all dates (point-in-time snapshot).
    """
    signal = get_insider_signal(ticker)
    return pd.DataFrame(
        {"insider_score": signal["insider_score"]},
        index=date_index,
    )
