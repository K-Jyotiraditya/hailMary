"""
Feature Engineer: downloads historical OHLCV and builds the training feature
matrix for XGBoost. Also provides live single-ticker feature extraction
for the pipeline.

Features per (stock, date):
  Technical  — RSI, MACD histogram, BB position, ATR%, volume ratio
  Momentum   — 5d / 21d / 63d returns, 12-1 month momentum factor
  Trend      — price vs SMA20 / SMA50
  Market     — VIX level, SPY 5d return, sector ETF 5d return

Label: did the stock close higher 5 trading days later? (1 = yes, 0 = no)
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.watchlist import WATCHLIST

SECTOR_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK",
    "GOOGL": "XLK", "META": "XLK", "NFLX": "XLK",
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY",
    "JPM": "XLF", "V": "XLF",
    "UNH": "XLV", "JNJ": "XLV",
    "XOM": "XLE", "PG": "XLP",
}

FORWARD_DAYS = 5

FEATURE_COLS = [
    # Technical
    "rsi", "macd_hist", "bb_pos", "atr_pct", "volume_ratio",
    # Momentum
    "ret_5d", "ret_21d", "ret_63d", "momentum_12_1",
    # Trend
    "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
    # Market
    "sector_ret_5d", "spy_ret_5d", "vix",
    # PEAD (earnings surprise)
    "earnings_surprise_pct", "pead_signal",
    # Google Trends
    "gtrends_spike_ratio", "gtrends_spike_flag",
    # Insider buying
    "insider_score",
]


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    lo = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / lo.replace(0, np.nan))


def _features_for(ticker: str, close_all: pd.DataFrame, volume_all: pd.DataFrame) -> pd.DataFrame:
    """Compute all features for one ticker from the pre-downloaded matrices."""
    if ticker not in close_all.columns:
        return pd.DataFrame()

    c = close_all[ticker].dropna()
    v = volume_all[ticker].dropna() if ticker in volume_all.columns else pd.Series(dtype=float)

    if len(c) < 210:
        return pd.DataFrame()

    sma20  = c.rolling(20).mean()
    sma50  = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    std20  = c.rolling(20).std()
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20

    vol_ratio = (v / v.rolling(20).mean()).reindex(c.index) if not v.empty else pd.Series(1.0, index=c.index)

    sector_etf = SECTOR_MAP.get(ticker, "SPY")
    sec_col = sector_etf if sector_etf in close_all.columns else "SPY"
    sec_ret  = close_all[sec_col].pct_change(5).reindex(c.index)
    spy_ret  = close_all["SPY"].pct_change(5).reindex(c.index)
    vix_lvl  = close_all["^VIX"].reindex(c.index) if "^VIX" in close_all.columns else pd.Series(20.0, index=c.index)

    fwd     = c.pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)
    spy_fwd = close_all["SPY"].pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS).reindex(c.index)
    # Relative return label: 1 if this stock outperforms SPY over next 5 days
    rel_fwd = fwd - spy_fwd

    df = pd.DataFrame({
        "rsi":              _rsi(c),
        "macd_hist":        macd - macd_sig,
        "bb_pos":           (c - bb_lower) / (bb_upper - bb_lower + 1e-9),
        "atr_pct":          c.diff().abs().rolling(14).mean() / c,
        "volume_ratio":     vol_ratio,
        "ret_5d":           c.pct_change(5),
        "ret_21d":          c.pct_change(21),
        "ret_63d":          c.pct_change(63),
        "momentum_12_1":    c.pct_change(252) - c.pct_change(21),
        "price_vs_sma20":   (c - sma20) / sma20,
        "price_vs_sma50":   (c - sma50) / sma50,
        "price_vs_sma200":  (c - sma200) / sma200.replace(0, np.nan),
        "sector_ret_5d":    sec_ret,
        "spy_ret_5d":       spy_ret,
        "vix":              vix_lvl,
        "fwd_ret":          fwd,
        "rel_fwd_ret":      rel_fwd,
        # Label: does this stock beat SPY over next 5 days?
        "label":            (rel_fwd > 0).astype(int),
    }, index=c.index)

    BASE_COLS = [
        "rsi", "macd_hist", "bb_pos", "atr_pct", "volume_ratio",
        "ret_5d", "ret_21d", "ret_63d", "momentum_12_1",
        "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
        "sector_ret_5d", "spy_ret_5d", "vix",
    ]
    df["ticker"] = ticker
    return df.dropna(subset=BASE_COLS + ["label"])


def build_features(period: str = "2y") -> pd.DataFrame:
    """Download history for all watchlist stocks and return feature + label matrix."""
    from data.earnings_tracker import build_pead_features
    from data.google_trends import build_trends_features

    sector_etfs = list(set(SECTOR_MAP.values()))
    all_tickers = list(set(WATCHLIST + sector_etfs + ["SPY", "^VIX"]))

    print(f"  Downloading {period} history for {len(all_tickers)} symbols...")
    raw = yf.download(all_tickers, period=period, auto_adjust=True,
                      progress=False, threads=True)

    if isinstance(raw.columns, pd.MultiIndex):
        close_all = raw["Close"]
        volume_all = raw["Volume"] if "Volume" in raw.columns.get_level_values(0) else pd.DataFrame()
    else:
        close_all = raw[["Close"]].rename(columns={"Close": all_tickers[0]})
        volume_all = pd.DataFrame()

    records = []
    for ticker in WATCHLIST:
        df = _features_for(ticker, close_all, volume_all)
        if df.empty:
            continue

        date_idx = df.index

        # ── PEAD features ─────────────────────────────────────────────────
        try:
            pead_df = build_pead_features(ticker, date_idx)
            df = df.join(pead_df[["earnings_surprise_pct", "pead_signal"]], how="left")
        except Exception as e:
            print(f"  [PEAD] {ticker}: {e}")
            df["earnings_surprise_pct"] = 0.0
            df["pead_signal"] = 0.0

        # ── Google Trends features ─────────────────────────────────────────
        try:
            trends_df = build_trends_features(ticker, date_idx)
            df = df.join(trends_df[["gtrends_spike_ratio", "gtrends_spike_flag"]], how="left")
        except Exception as e:
            print(f"  [GTrends] {ticker}: {e}")
            df["gtrends_spike_ratio"] = 1.0
            df["gtrends_spike_flag"]  = 0.0

        # ── Insider buying features ────────────────────────────────────────
        try:
            from data.insider_buying import build_insider_features
            insider_df = build_insider_features(ticker, date_idx)
            df = df.join(insider_df[["insider_score"]], how="left")
        except Exception as e:
            print(f"  [Insider] {ticker}: {e}")
            df["insider_score"] = 0.0

        df[["earnings_surprise_pct", "pead_signal"]] = \
            df[["earnings_surprise_pct", "pead_signal"]].fillna(0.0)
        df[["gtrends_spike_ratio", "gtrends_spike_flag"]] = \
            df[["gtrends_spike_ratio", "gtrends_spike_flag"]].fillna({"gtrends_spike_ratio": 1.0,
                                                                       "gtrends_spike_flag": 0.0})
        df["insider_score"] = df["insider_score"].fillna(0.0)
        records.append(df)

    if not records:
        raise ValueError("No feature data could be built — check yfinance connectivity.")

    result = pd.concat(records)
    result.index.name = "date"
    result = result.reset_index()
    print(f"  Built {len(result)} samples across {result['ticker'].nunique()} stocks "
          f"with {len(FEATURE_COLS)} features.")
    return result


def get_live_features(ticker: str) -> dict | None:
    """
    Compute today's full feature vector for a single ticker.
    Includes technical + PEAD + Google Trends features.
    Returns a dict keyed by FEATURE_COLS, or None on failure.
    """
    from data.earnings_tracker import get_live_pead
    from data.google_trends import get_live_trends

    sector_etf = SECTOR_MAP.get(ticker, "SPY")
    needed = list({ticker, sector_etf, "SPY", "^VIX"})

    try:
        raw = yf.download(needed, period="2y", auto_adjust=True,
                          progress=False, threads=False)
    except Exception:
        return None

    if raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        close_all = raw["Close"]
        volume_all = raw["Volume"] if "Volume" in raw.columns.get_level_values(0) else pd.DataFrame()
    else:
        return None

    df = _features_for(ticker, close_all, volume_all)
    if df.empty:
        return None

    row = df.iloc[-1]
    base = {col: float(row[col]) for col in [
        "rsi", "macd_hist", "bb_pos", "atr_pct", "volume_ratio",
        "ret_5d", "ret_21d", "ret_63d", "momentum_12_1",
        "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
        "sector_ret_5d", "spy_ret_5d", "vix",
    ]}

    # PEAD
    try:
        pead = get_live_pead(ticker)
        base["earnings_surprise_pct"] = pead.get("earnings_surprise_pct", 0.0)
        base["pead_signal"]           = pead.get("pead_signal", 0.0)
    except Exception:
        base["earnings_surprise_pct"] = 0.0
        base["pead_signal"]           = 0.0

    # Google Trends
    try:
        trends = get_live_trends(ticker)
        base["gtrends_spike_ratio"] = trends.get("gtrends_spike_ratio", 1.0)
        base["gtrends_spike_flag"]  = trends.get("gtrends_spike_flag", 0.0)
    except Exception:
        base["gtrends_spike_ratio"] = 1.0
        base["gtrends_spike_flag"]  = 0.0

    # Insider buying
    try:
        from data.insider_buying import get_insider_signal
        insider = get_insider_signal(ticker)
        base["insider_score"] = insider.get("insider_score", 0.0)
    except Exception:
        base["insider_score"] = 0.0

    return base
