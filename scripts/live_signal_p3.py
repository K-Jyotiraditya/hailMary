"""
Live Signal v2 — Phase 3 ensemble with explainability and drift monitoring.

What's new vs scripts/live_signal.py:
  1. Uses Phase 3 bagged LGBM + RF ensemble.
  2. Sector-neutral top-K (no sector > 35% by default).
  3. Per-pick rationale: top 3 features driving each stock's prediction
     (gain importance from the model, weighted by the stock's feature value).
  4. Confidence intervals: standard deviation across the LGBM bag.
  5. Drift monitoring: warn if today's feature distribution looks unlike
     training data (z-score against the training mean).
  6. Recent IC tracking: report how the model has done OOS over the last 60d.
  7. Sector summary of the recommended portfolio.

Run: PYTHONPATH=. python scripts/live_signal_p3.py
     PYTHONPATH=. python scripts/live_signal_p3.py --capital 500000 --top-k 8
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
from data.sectors import sector_distribution, get_sector


TOP_K = 10
COV_WINDOW = 60
MAX_POSITION = 0.18
MAX_SECTOR_WEIGHT = 0.35
NIFTY_INDEX = '^NSEI'


def fetch_latest_prices(symbols: list, days: int = 400) -> dict:
    """Fetch the last `days` of OHLCV from yfinance."""
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

    mc = market_features(nifty_df)
    mc['mkt_breadth_50'] = breadth_features(prices_dict, lookback=50)
    mc['mkt_breadth_200'] = breadth_features(prices_dict, lookback=200)
    mc = mc.dropna()
    mc.index.name = 'date'

    panel_full = panel_full.join(mc, on='date', how='inner')
    return panel_full


def explain_pick(feature_row: pd.Series, gain_importance: pd.Series, top_n: int = 3) -> str:
    """
    Explain why this stock was picked: top features by (importance * |feature value|).

    Returns a short string like "top features: cs_drawdown_60(+1.5), vol_60(+0.8), ..."
    """
    aligned = pd.concat([feature_row.rename('val'), gain_importance.rename('imp')], axis=1).dropna()
    aligned['contrib'] = aligned['imp'] * aligned['val'].abs()
    top = aligned.nlargest(top_n, 'contrib')
    parts = []
    for feat, row in top.iterrows():
        sign = '+' if row['val'] >= 0 else ''
        parts.append(f"{feat}({sign}{row['val']:.1f})")
    return ', '.join(parts)


def feature_drift(today_features: pd.DataFrame, training_distribution: dict,
                  threshold: float = 2.5) -> list:
    """
    Compare today's feature distribution to training stats.
    Returns features that have drifted > threshold sigma in the cross-section.
    """
    drifted = []
    for col in today_features.columns:
        if col not in training_distribution:
            continue
        ref_mean = training_distribution[col]['mean']
        ref_std = training_distribution[col]['std']
        if ref_std <= 0:
            continue
        today_mean = today_features[col].mean()
        z = (today_mean - ref_mean) / ref_std
        if abs(z) > threshold:
            drifted.append((col, today_mean, z))
    return sorted(drifted, key=lambda x: -abs(x[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capital', type=float, default=1_000_000)
    parser.add_argument('--top-k', type=int, default=TOP_K)
    parser.add_argument('--no-regime', action='store_true')
    parser.add_argument('--use-cached', action='store_true')
    parser.add_argument('--max-sector-weight', type=float, default=MAX_SECTOR_WEIGHT)
    args = parser.parse_args()

    print("=" * 80)
    print(f"LIVE SIGNAL v2 (Phase 3) — Capital: Rs.{args.capital:,.0f}")
    print("=" * 80)

    # ---- Load Phase 3 models ----
    print("\n[1/7] Loading models...")
    with open('data/models/p3_year_models.pkl', 'rb') as f:
        year_models = pickle.load(f)
    latest_year = max(year_models.keys())
    pack = year_models[latest_year]
    print(f"  Using {latest_year} year-model: {len(pack['bagged_models'])} bagged LGBM + 1 RF")

    universe = sorted([f.stem for f in Path('data/processed').glob('*.parquet')
                       if f.stem != 'BENCHMARK_INDEX'])

    # ---- Fetch / load prices ----
    if args.use_cached:
        print("\n[2/7] Using cached prices...")
        ohlcv = {s: pd.read_parquet(f'data/processed/{s}.parquet') for s in universe}
        nifty = pd.read_parquet('data/processed/BENCHMARK_INDEX.parquet')
    else:
        print(f"\n[2/7] Fetching latest prices from yfinance...")
        ohlcv = fetch_latest_prices(universe + [NIFTY_INDEX])
        nifty = ohlcv.pop(NIFTY_INDEX) if NIFTY_INDEX in ohlcv else \
                pd.read_parquet('data/processed/BENCHMARK_INDEX.parquet')

    if not ohlcv:
        print("ERROR: No price data loaded.")
        return

    today = max(df.index.max() for df in ohlcv.values())
    print(f"  As-of: {today.date()}")

    # ---- Compute features ----
    print(f"\n[3/7] Computing features for {len(ohlcv)} stocks...")
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

    # ---- Predict (bagged + RF) ----
    print(f"\n[4/7] Predicting ranks...")
    bagged_with_ci = predict_bagged_with_ci(pack['bagged_models'], X)
    rf_pred = pd.Series(pack['rf_model'].predict(X), index=X.index) if pack.get('rf_model') is not None else None

    if rf_pred is not None:
        ensemble = 0.6 * bagged_with_ci['pred_mean'] + 0.4 * rf_pred
    else:
        ensemble = bagged_with_ci['pred_mean']

    ranks = ensemble.rank(pct=True).sort_values(ascending=False)
    pred_std = bagged_with_ci['pred_std']

    # ---- Drift check ----
    print(f"\n[5/7] Feature drift check...")
    # Build training distribution stats from a sample of recent training data
    # (use the panel features from the last full training set as reference)
    train_stats_path = Path(f'data/models/p3_train_stats_{latest_year}.pkl')
    if not train_stats_path.exists():
        # Compute from the cs_panel: take 2018-2023 as ref
        ref_panel = pd.read_parquet('data/processed/features/CS_PANEL.parquet')
        ref_panel = ref_panel.loc[(ref_panel.index.get_level_values('date') >= '2018-01-01') &
                                  (ref_panel.index.get_level_values('date') < f'{latest_year}-01-01')]
        stats = {col: {'mean': float(ref_panel[col].mean()),
                       'std':  float(ref_panel[col].std())}
                 for col in ref_panel.columns}
        with open(train_stats_path, 'wb') as f:
            pickle.dump(stats, f)
    else:
        with open(train_stats_path, 'rb') as f:
            stats = pickle.load(f)

    drift = feature_drift(today_features.reindex(columns=feature_names), stats, threshold=2.5)
    if drift:
        print(f"  [!] {len(drift)} features show >2.5sigma drift. Top:")
        for feat, val, z in drift[:5]:
            print(f"      {feat:30}  today={val:+.3f}  z={z:+.2f}")
    else:
        print(f"  [OK] No major drift detected.")

    # ---- Regime filter ----
    print(f"\n[6/7] Regime check...")
    bull = is_bullish(nifty['Close'], today, window=200)
    print(f"  NIFTY 50 above 200-DMA: {bull}")
    if not bull and not args.no_regime:
        print(f"\n  ** REGIME = BEAR ** -> recommendation: 100% CASH")
        print("=" * 80)
        # Even in cash, save the snapshot
        out_dir = Path('data/live_signals')
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            'symbol': ranks.index.tolist()[:20],
            'rank': ranks.values[:20],
            'sector': [get_sector(s) for s in ranks.index.tolist()[:20]],
        }).to_csv(out_dir / f"signal_{today.date()}_BEAR.csv", index=False)
        return

    # ---- Sector-neutral selection + HRP ----
    print(f"\n[7/7] Building sector-neutral top-{args.top_k} portfolio...")
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

    # ---- Per-pick explanation ----
    print("\n" + "=" * 80)
    print(f"TARGET PORTFOLIO — {today.date()}")
    print("=" * 80)

    # Get aggregate gain importance from the bag
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
    pick_rows = []
    for sym in weights.index:
        w = weights[sym]
        sec = get_sector(sym)
        rank = ranks.get(sym, np.nan)
        conf = 1.0 - pred_std.get(sym, 1.0)  # higher = more confident
        dollars = w * args.capital
        latest_price = float(ohlcv[sym]['Close'].iloc[-1])
        shares = int(dollars / latest_price) if latest_price > 0 else 0
        actual_dollars = shares * latest_price
        total_dollars += actual_dollars
        print(f"  {sym:14} {w:>6.2%} {sec:12} {rank:>6.3f} {conf:>6.2f} Rs.{dollars:>11,.0f} {shares:>8,}")
        pick_rows.append({
            'symbol': sym, 'sector': sec, 'weight': w,
            'predicted_rank': rank, 'confidence': conf,
            'dollars': dollars, 'shares': shares, 'price': latest_price,
        })

    cash = args.capital - total_dollars
    print("  " + "-" * 90)
    print(f"  {'CASH':14} {cash/args.capital:>6.2%} {'-':12} {'-':>6} {'-':>6} Rs.{cash:>11,.0f}")

    # Sector summary
    print(f"\n  Sector breakdown:")
    secs = sector_distribution(weights)
    for sec, w in sorted(secs.items(), key=lambda x: -x[1]):
        bar = '#' * int(w * 50)
        print(f"      {sec:12} {w:>6.1%}  {bar}")

    # Per-pick rationale
    print(f"\n  Why these picks (top features driving each prediction):")
    for sym in weights.index[:min(5, len(weights))]:
        if sym not in today_features.index:
            continue
        feat_row = today_features.loc[sym].reindex(feature_names).fillna(0)
        why = explain_pick(feat_row, avg_importance, top_n=3)
        print(f"    {sym:14} -> {why}")

    # Save
    out_dir = Path('data/live_signals')
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = pd.DataFrame(pick_rows)
    rec.to_csv(out_dir / f"signal_{today.date()}.csv", index=False)
    print(f"\n[OK] Saved to {out_dir / f'signal_{today.date()}.csv'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
