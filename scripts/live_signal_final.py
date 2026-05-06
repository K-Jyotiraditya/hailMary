"""
Final Live Signal Generator — Phase 5 with three production presets.

Choose your tradeoff:
  --mode calmar     : Calmar champion. MaxDD -11%, Sharpe 1.26.
                      Tuned LGBM + composite regime (200-DMA + 50-DMA + 60d momentum + vol calm).
                      Use when drawdown survival matters more than peak return.

  --mode sharpe     : Sharpe champion. MaxDD -20%, Sharpe 1.38.
                      Tuned LGBM + simple SMA-200 regime.
                      Use when you can stomach -20% to chase max risk-adjusted return.

  --mode balanced   : Phase 3 default. MaxDD -16.5%, Sharpe 1.33. (DEFAULT)
                      Default LGBM + simple SMA-200 regime.

All modes: bagged LGBM (5 seeds) + RF ensemble, sector-neutral top-K.

Run: PYTHONPATH=. python scripts/live_signal_final.py
     PYTHONPATH=. python scripts/live_signal_final.py --mode calmar --capital 500000
     PYTHONPATH=. python scripts/live_signal_final.py --mode sharpe --capital 1000000
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
from ml.bagging import predict_bagged_with_ci
from portfolio.sector_neutral import sector_neutral_top_k
from portfolio.regime_filter import is_bullish
from portfolio.regime_filter_v2 import is_bullish_v2
from data.sectors import sector_distribution, get_sector


MODE_CONFIGS = {
    'balanced': {
        'description': 'Phase 3 — default LGBM + SMA-200 regime',
        'models_path': 'data/models/p3_year_models.pkl',
        'expected': {'sharpe': 1.33, 'maxdd': -0.165, 'cagr': 0.174},
        'regime_fn': lambda nifty, date: is_bullish(nifty, date, window=200),
    },
    'sharpe': {
        'description': 'P5 Sharpe champion — tuned LGBM + SMA-200 regime',
        'models_path': 'data/models/p5_year_models.pkl',
        'expected': {'sharpe': 1.38, 'maxdd': -0.206, 'cagr': 0.186},
        'regime_fn': lambda nifty, date: is_bullish(nifty, date, window=200),
    },
    'calmar': {
        'description': 'P5 Calmar champion — tuned LGBM + composite regime',
        'models_path': 'data/models/p5_year_models.pkl',
        'expected': {'sharpe': 1.26, 'maxdd': -0.111, 'cagr': 0.145},
        'regime_fn': lambda nifty, date: is_bullish_v2(
            nifty, date,
            require_sma200=True, require_sma50=True,
            require_momentum_60d=True, require_vol_calm=True,
            vol_ratio_max=1.6,
        ),
    },
}


TOP_K = 10
COV_WINDOW = 60
MAX_POSITION = 0.18
MAX_SECTOR_WEIGHT = 0.35
NIFTY_INDEX = '^NSEI'


def fetch_latest_prices(symbols: list, days: int = 400) -> dict:
    end = datetime.now()
    start = end - pd.Timedelta(days=days * 2)
    out = {}
    for s in symbols:
        try:
            ticker = f'{s}.NS' if not s.startswith('^') else s
            df = yf.download(ticker, start=start, end=end, progress=False, timeout=10)
            if df.empty or len(df) < 250:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
            out[s] = df
        except Exception:
            pass
    return out


def compute_per_stock_features(ohlcv_dict: dict) -> dict:
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


def build_today_panel(per_stock_features, nifty_df, prices_dict):
    panel = stack_to_panel(per_stock_features)
    cs = cross_sectional_rank(panel, prefix='cs_')
    panel_full = pd.concat([panel, cs], axis=1)
    mc = market_features(nifty_df)
    mc['mkt_breadth_50'] = breadth_features(prices_dict, lookback=50)
    mc['mkt_breadth_200'] = breadth_features(prices_dict, lookback=200)
    mc = mc.dropna()
    mc.index.name = 'date'
    panel_full = panel_full.join(mc, on='date', how='inner')
    return panel_full


def explain_pick(feature_row, gain_importance, top_n=3) -> str:
    aligned = pd.concat([feature_row.rename('val'), gain_importance.rename('imp')], axis=1).dropna()
    aligned['contrib'] = aligned['imp'] * aligned['val'].abs()
    top = aligned.nlargest(top_n, 'contrib')
    parts = []
    for feat, row in top.iterrows():
        sign = '+' if row['val'] >= 0 else ''
        parts.append(f"{feat}({sign}{row['val']:.1f})")
    return ', '.join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=list(MODE_CONFIGS.keys()), default='balanced',
                        help='Production preset (calmar/sharpe/balanced)')
    parser.add_argument('--capital', type=float, default=1_000_000)
    parser.add_argument('--top-k', type=int, default=TOP_K)
    parser.add_argument('--no-regime', action='store_true')
    parser.add_argument('--use-cached', action='store_true')
    parser.add_argument('--max-sector-weight', type=float, default=MAX_SECTOR_WEIGHT)
    args = parser.parse_args()

    cfg = MODE_CONFIGS[args.mode]

    print("=" * 80)
    print(f"LIVE SIGNAL FINAL — mode: {args.mode.upper()}")
    print("=" * 80)
    print(f"  {cfg['description']}")
    print(f"  Backtested: Sharpe={cfg['expected']['sharpe']:.2f} | "
          f"CAGR={cfg['expected']['cagr']:.1%} | MaxDD={cfg['expected']['maxdd']:.1%}")

    # ---- Load models ----
    print(f"\n[1/7] Loading {args.mode} models...")
    with open(cfg['models_path'], 'rb') as f:
        year_models = pickle.load(f)
    latest_year = max(year_models.keys())
    pack = year_models[latest_year]
    print(f"  Using {latest_year} year-model: {len(pack['bagged_models'])} bagged LGBM + 1 RF")

    universe = sorted([f.stem for f in Path('data/processed').glob('*.parquet')
                       if f.stem != 'BENCHMARK_INDEX'])

    # ---- Fetch prices ----
    if args.use_cached:
        print("\n[2/7] Using cached prices...")
        ohlcv = {s: pd.read_parquet(f'data/processed/{s}.parquet') for s in universe}
        nifty = pd.read_parquet('data/processed/BENCHMARK_INDEX.parquet')
    else:
        print(f"\n[2/7] Fetching latest prices...")
        ohlcv = fetch_latest_prices(universe + [NIFTY_INDEX])
        nifty = ohlcv.pop(NIFTY_INDEX) if NIFTY_INDEX in ohlcv else \
                pd.read_parquet('data/processed/BENCHMARK_INDEX.parquet')

    today = max(df.index.max() for df in ohlcv.values())
    print(f"  As-of: {today.date()}")

    # ---- Features ----
    print(f"\n[3/7] Computing features...")
    per_stock = compute_per_stock_features(ohlcv)
    prices_dict = {s: df['Close'] for s, df in ohlcv.items()}
    panel_today = build_today_panel(per_stock, nifty, prices_dict)
    lagged = panel_today.groupby(level='symbol').shift(1).dropna()
    if today in lagged.index.get_level_values('date'):
        today_features = lagged.loc[today]
    else:
        latest_date = lagged.index.get_level_values('date').max()
        print(f"  Today's features not yet computable; using {latest_date.date()}")
        today_features = lagged.loc[latest_date]

    feature_names = pack['feature_names']
    X = today_features.reindex(columns=feature_names).fillna(0.0)

    # ---- Predict ----
    print(f"\n[4/7] Predicting ranks...")
    bagged_with_ci = predict_bagged_with_ci(pack['bagged_models'], X)
    rf_pred = pd.Series(pack['rf_model'].predict(X), index=X.index) \
              if pack.get('rf_model') is not None else None
    if rf_pred is not None:
        ensemble = 0.6 * bagged_with_ci['pred_mean'] + 0.4 * rf_pred
    else:
        ensemble = bagged_with_ci['pred_mean']
    ranks = ensemble.rank(pct=True).sort_values(ascending=False)

    # ---- Regime ----
    print(f"\n[5/7] Regime check ({args.mode} filter)...")
    bull = cfg['regime_fn'](nifty['Close'], today)
    print(f"  Regime bullish: {bull}")
    if not bull and not args.no_regime:
        print(f"\n  ** REGIME = BEAR ** -> recommendation: 100% CASH")
        # Save snapshot
        out_dir = Path('data/live_signals')
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            'symbol': ranks.index.tolist()[:20],
            'rank': ranks.values[:20],
            'sector': [get_sector(s) for s in ranks.index.tolist()[:20]],
        }).to_csv(out_dir / f"signal_{args.mode}_{today.date()}_BEAR.csv", index=False)
        print("=" * 80)
        return

    # ---- Sector-neutral selection ----
    print(f"\n[6/7] Building sector-neutral top-{args.top_k} portfolio...")
    returns_df = pd.DataFrame({s: ohlcv[s]['Close'].pct_change() for s in universe
                                if s in ohlcv})
    weights = sector_neutral_top_k(
        ranks, returns_df.tail(COV_WINDOW),
        k=args.top_k,
        max_position=MAX_POSITION,
        max_sector_weight=args.max_sector_weight,
        min_sectors=4,
    )
    weights = weights[weights > 0.001].sort_values(ascending=False)

    # ---- Output portfolio ----
    print("\n" + "=" * 80)
    print(f"TARGET PORTFOLIO ({args.mode}) — {today.date()}  Capital: Rs.{args.capital:,.0f}")
    print("=" * 80)

    importances = []
    for m in pack['bagged_models']:
        importances.append(pd.Series(
            m.booster_.feature_importance(importance_type='gain'),
            index=feature_names,
        ))
    avg_importance = sum(importances) / len(importances)

    print(f"\n  {'Symbol':14} {'Weight':>7} {'Sector':12} {'Pred':>6} {'Conf':>6} {'Dollars':>14} {'Shares':>8}")
    print("  " + "-" * 90)
    total_dollars = 0
    rows = []
    for sym in weights.index:
        w = weights[sym]
        sec = get_sector(sym)
        rank = ranks.get(sym, np.nan)
        pred_std_val = bagged_with_ci['pred_std'].get(sym, 1.0)
        conf = 1.0 - pred_std_val
        dollars = w * args.capital
        latest_price = float(ohlcv[sym]['Close'].iloc[-1])
        shares = int(dollars / latest_price) if latest_price > 0 else 0
        actual_dollars = shares * latest_price
        total_dollars += actual_dollars
        print(f"  {sym:14} {w:>6.2%} {sec:12} {rank:>6.3f} {conf:>6.2f} Rs.{dollars:>11,.0f} {shares:>8,}")
        rows.append({'symbol': sym, 'sector': sec, 'weight': w, 'rank': rank,
                     'confidence': conf, 'dollars': dollars, 'shares': shares,
                     'price': latest_price})

    cash = args.capital - total_dollars
    print("  " + "-" * 90)
    print(f"  {'CASH':14} {cash/args.capital:>6.2%} {'-':12} {'-':>6} {'-':>6} Rs.{cash:>11,.0f}")

    secs = sector_distribution(weights)
    print(f"\n  Sector breakdown:")
    for sec, w in sorted(secs.items(), key=lambda x: -x[1]):
        bar = '#' * int(w * 50)
        print(f"      {sec:12} {w:>6.1%}  {bar}")

    print(f"\n  Top picks rationale:")
    for sym in weights.index[:min(5, len(weights))]:
        if sym not in today_features.index:
            continue
        feat_row = today_features.loc[sym].reindex(feature_names).fillna(0)
        why = explain_pick(feat_row, avg_importance, top_n=3)
        print(f"    {sym:14} -> {why}")

    # Save
    print(f"\n[7/7] Saving signal...")
    out_dir = Path('data/live_signals')
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / f"signal_{args.mode}_{today.date()}.csv", index=False)
    print(f"  -> {out_dir / f'signal_{args.mode}_{today.date()}.csv'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
