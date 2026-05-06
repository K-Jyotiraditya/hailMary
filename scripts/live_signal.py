"""
Live Signal Generator — produce TODAY's portfolio recommendation.

Workflow on a daily basis:
  1. Pull latest OHLCV from yfinance for the universe + NIFTY 50 index.
  2. Recompute features (technical + price-based + cross-sectional + market).
  3. Use the most recent year-model to predict ranks for each stock.
  4. Apply the same selection logic as the backtest: top-K + HRP + regime
     filter (no vol-targeting in production by default).
  5. Print a clean target portfolio: which stocks, what weights, what action.

This is the bridge from research to trading. Run this every morning before
market open and execute the trades it produces.

Run: PYTHONPATH=. python scripts/live_signal.py
     PYTHONPATH=. python scripts/live_signal.py --capital 500000
"""
import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf

from features.technical import compute_all_technical
from features.price_based import compute_all_price_features
from features.ml_features import handle_nans, winsorize_features, rolling_zscore
from features.cross_sectional import stack_to_panel, cross_sectional_rank
from features.market_context import market_features, breadth_features
from ml.ensemble import ensemble_predict
from portfolio.optimizer import hrp_weights_robust
from portfolio.regime_filter import is_bullish


TOP_K = 10
COV_WINDOW = 60
MAX_POSITION = 0.18
NIFTY_INDEX = '^NSEI'


def fetch_latest_prices(symbols: list, days: int = 400) -> dict:
    """Fetch the last `days` of OHLCV from yfinance for each symbol."""
    end = datetime.now()
    start = end - pd.Timedelta(days=days * 2)  # buffer for non-trading days

    out = {}
    for s in symbols:
        try:
            ticker = f'{s}.NS' if not s.startswith('^') else s
            df = yf.download(ticker, start=start, end=end, progress=False, timeout=10)
            if df.empty or len(df) < 250:
                print(f"  [skip] {s}: only {len(df)} rows")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
            out[s] = df
        except Exception as e:
            print(f"  [err] {s}: {str(e)[:50]}")
    return out


def compute_per_stock_features(ohlcv_dict: dict) -> dict:
    """Re-engineer features per stock identical to the offline pipeline."""
    out = {}
    for symbol, df in ohlcv_dict.items():
        tech = compute_all_technical(df)
        price = compute_all_price_features(df)
        raw = pd.concat([tech, price], axis=1)
        raw = handle_nans(raw, max_nan_pct=0.10)
        ml = winsorize_features(raw, lower_pct=0.005, upper_pct=0.995)
        ml = rolling_zscore(ml, window=252)
        ml = ml.dropna()
        if len(ml) > 0:
            out[symbol] = ml
    return out


def build_today_panel(per_stock_features: dict, nifty_df: pd.DataFrame, prices_dict: dict):
    """Stack features into a panel and add cross-sectional + market context."""
    panel = stack_to_panel(per_stock_features)
    cs = cross_sectional_rank(panel, prefix='cs_')
    panel_full = pd.concat([panel, cs], axis=1)

    # Market context
    mc = market_features(nifty_df)
    mc['mkt_breadth_50'] = breadth_features(prices_dict, lookback=50)
    mc['mkt_breadth_200'] = breadth_features(prices_dict, lookback=200)
    mc = mc.dropna()
    mc.index.name = 'date'

    panel_full = panel_full.join(mc, on='date', how='inner')
    return panel_full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capital', type=float, default=1_000_000,
                        help='Total capital for sizing dollar allocations')
    parser.add_argument('--top-k', type=int, default=TOP_K)
    parser.add_argument('--no-regime', action='store_true', help='Disable regime filter')
    parser.add_argument('--use-cached', action='store_true',
                        help='Use cached prices in data/processed/ instead of fresh download')
    args = parser.parse_args()

    print("=" * 75)
    print("LIVE SIGNAL — TODAY'S PORTFOLIO RECOMMENDATION")
    print("=" * 75)

    # ---- Load Phase 2.1 models ----
    print("\n[1/6] Loading models...")
    with open('data/models/p21_year_models.pkl', 'rb') as f:
        year_models = pickle.load(f)
    latest_year = max(year_models.keys())
    model_pack = year_models[latest_year]
    print(f"  Using {latest_year} year-model")
    print(f"  Features: {len(model_pack['feature_names'])}")

    # ---- Determine universe ----
    universe = sorted([f.stem for f in Path('data/processed').glob('*.parquet')
                       if f.stem != 'BENCHMARK_INDEX'])
    print(f"  Universe: {len(universe)} stocks")

    # ---- Fetch / load prices ----
    if args.use_cached:
        print("\n[2/6] Using cached prices (data/processed/)...")
        ohlcv = {}
        for s in universe:
            ohlcv[s] = pd.read_parquet(f'data/processed/{s}.parquet')
        nifty = pd.read_parquet('data/processed/BENCHMARK_INDEX.parquet')
    else:
        print(f"\n[2/6] Fetching latest prices from yfinance...")
        ohlcv = fetch_latest_prices(universe + [NIFTY_INDEX])
        nifty = ohlcv.pop(NIFTY_INDEX) if NIFTY_INDEX in ohlcv else None
        if nifty is None:
            print("  Warning: NIFTY index fetch failed, using cached")
            nifty = pd.read_parquet('data/processed/BENCHMARK_INDEX.parquet')

    if not ohlcv:
        print("ERROR: No price data loaded.")
        return

    today = max(df.index.max() for df in ohlcv.values())
    print(f"  As-of date: {today.date()}")

    # ---- Compute features ----
    print(f"\n[3/6] Computing features for {len(ohlcv)} stocks...")
    per_stock = compute_per_stock_features(ohlcv)
    print(f"  Features computed: {len(per_stock)} stocks")

    prices_dict = {s: df['Close'] for s, df in ohlcv.items()}
    panel_today = build_today_panel(per_stock, nifty, prices_dict)
    print(f"  Panel: {panel_today.shape}")

    # ---- Get features for today ----
    # Lag by 1 day per symbol — features available at end of yesterday predict today's entry
    lagged = panel_today.groupby(level='symbol').shift(1).dropna()
    if today in lagged.index.get_level_values('date'):
        today_features = lagged.loc[today]
    else:
        # Fall back to most recent date with features
        latest_date = lagged.index.get_level_values('date').max()
        print(f"  Today's features not yet computable; using {latest_date.date()}")
        today_features = lagged.loc[latest_date]

    # ---- Predict ----
    print(f"\n[4/6] Predicting ranks...")
    feature_names = model_pack['feature_names']
    X = today_features.reindex(columns=feature_names).fillna(0.0)
    lgbm_pred = pd.Series(model_pack['lgbm'].predict(X), index=X.index)
    rf_pred = pd.Series(model_pack['rf'].predict(X), index=X.index)
    ensemble_raw = 0.6 * lgbm_pred + 0.4 * rf_pred
    ranks = ensemble_raw.rank(pct=True).sort_values(ascending=False)

    print(f"\n  Top-{args.top_k} stocks by predicted rank:")
    for i, (sym, r) in enumerate(ranks.head(args.top_k).items(), 1):
        print(f"    {i:>2}. {sym:14}  rank={r:.4f}")

    # ---- Regime filter ----
    print(f"\n[5/6] Regime check...")
    bull = is_bullish(nifty['Close'], today, window=200)
    print(f"  NIFTY 50 above 200-DMA: {bull}")
    if not bull and not args.no_regime:
        print("\n  ** REGIME = BEAR ** — recommendation: 100% CASH")
        print("=" * 75)
        return

    # ---- HRP weighting ----
    print(f"\n[6/6] HRP weighting and final portfolio...")
    top_syms = ranks.head(args.top_k).index.tolist()
    returns_df = pd.DataFrame({s: ohlcv[s]['Close'].pct_change() for s in top_syms})
    ret_window = returns_df.tail(COV_WINDOW).dropna(how='any')

    if len(ret_window) >= 30:
        try:
            hrp_w = hrp_weights_robust(ret_window)
        except Exception:
            hrp_w = pd.Series(1.0 / len(top_syms), index=top_syms)
    else:
        hrp_w = pd.Series(1.0 / len(top_syms), index=top_syms)

    hrp_w = hrp_w.clip(upper=MAX_POSITION)
    hrp_w = hrp_w / hrp_w.sum()

    # ---- Final output ----
    print("\n" + "=" * 75)
    print(f"TARGET PORTFOLIO — {today.date()}  Capital: Rs.{args.capital:,.0f}")
    print("=" * 75)
    print(f"\n  {'Symbol':14} {'Weight':>8} {'Dollars':>14} {'Latest Close':>14} {'Shares':>10}")
    print("  " + "-" * 60)
    total_dollars = 0
    for sym in top_syms:
        w = hrp_w[sym]
        dollars = w * args.capital
        latest_price = float(ohlcv[sym]['Close'].iloc[-1])
        shares = int(dollars / latest_price) if latest_price > 0 else 0
        actual_dollars = shares * latest_price
        total_dollars += actual_dollars
        print(f"  {sym:14} {w:>7.2%} Rs.{dollars:>11,.0f} Rs.{latest_price:>11,.2f} {shares:>10,}")
    cash = args.capital - total_dollars
    print("  " + "-" * 60)
    print(f"  {'CASH':14} {cash/args.capital:>7.2%} Rs.{cash:>11,.0f}")
    print(f"  {'TOTAL':14} {'100.00%':>8} Rs.{args.capital:>11,.0f}")

    print("\nNotes:")
    print("  - Execute as market-on-close to match backtest assumption")
    print("  - Realistic costs: ~5 bps commission, ~1 bps slippage built into backtest")
    print("  - Rebalance monthly: re-run this script on the first trading day of each month")

    # Save the recommendation
    out_dir = Path('data/live_signals')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"signal_{today.date()}.csv"
    rec = pd.DataFrame({
        'symbol': top_syms,
        'predicted_rank': [ranks[s] for s in top_syms],
        'weight': [hrp_w[s] for s in top_syms],
        'dollars': [hrp_w[s] * args.capital for s in top_syms],
        'latest_close': [float(ohlcv[s]['Close'].iloc[-1]) for s in top_syms],
    })
    rec.to_csv(out_path, index=False)
    print(f"\n[OK] Saved to {out_path}")
    print("=" * 75)


if __name__ == '__main__':
    main()
