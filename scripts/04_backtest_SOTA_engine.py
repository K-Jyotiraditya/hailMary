"""
Phase 5 FINAL — production-optimal config.

Based on the ablation study:
  - Tuned hyperparameters (Optuna): +0.05 Sharpe over defaults
  - Multi-horizon: HURTS, removed
  - Feature selection: marginal Sharpe loss + worse DD, removed
  - Composite regime filter: NEW, tightens cash-when-uncertain rule
  - Sector neutrality: kept (free risk reduction)
  - Bagged LGBM (5 seeds) + RF ensemble: kept

Run: PYTHONPATH=. python scripts/p5_final.py
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

from ml.cs_model import (
    assemble_panel_features, align_panel_features_labels, evaluate_predictions,
)
from ml.bagging import predict_bagged
from labels.rank_label import build_label_panel
from portfolio.regime_filter import sma_regime
from portfolio.regime_filter_v2 import composite_bullish_signal
from portfolio.rebalancer import get_rebalance_dates
from portfolio.sector_neutral import sector_neutral_top_k
from data.sectors import sector_distribution
from backtest.engine import run_backtest, BacktestConfig, equal_weight_signal
from backtest.results import compute_metrics, print_metrics, compare_strategies
from backtest.visualize import (
    plot_equity_curves, plot_drawdown, plot_monthly_returns, plot_rolling_sharpe,
)


# Config (informed by ablation) - Adjusted for mini 8-stock global stress tests
TOP_K = 3
COV_WINDOW = 60
MAX_POSITION = 0.35
MAX_SECTOR_WEIGHT = 1.0 # disabled sector constraints for mini-test
MIN_SECTORS = 1
N_BAGGING_SEEDS = 5
BACKTEST_START = '2019-01-01'


def make_lgbm(params, n_est):
    base = {'objective':'regression', 'metric':'rmse',
            'verbosity':-1, 'random_state':42, 'n_jobs':-1}
    return lgb.LGBMRegressor(n_estimators=n_est, **{**base, **params})


def make_rf():
    return RandomForestRegressor(
        n_estimators=200, max_depth=12,
        min_samples_split=100, min_samples_leaf=50,
        max_features=0.5, n_jobs=-1, random_state=42)


def train_bagged_with_seeds(X, y, params, n_est, n_seeds=N_BAGGING_SEEDS):
    """Multi-seed bagged training."""
    models = []
    for i in range(n_seeds):
        p = dict(params)
        p['random_state'] = 42 + i
        models.append(make_lgbm(p, n_est).fit(X, y))
    return models


def predict_panel_walk_forward(X, year_models):
    """Generate per-year predictions from year_models."""
    dates = X.index.get_level_values('date')
    pred = pd.Series(np.nan, index=X.index)
    for year in sorted(year_models.keys()):
        m = year_models[year]
        mask = (dates.year == year)
        if not mask.any():
            continue
        X_year = X[mask].reindex(columns=m['feature_names'])
        bagged = predict_bagged(m['bagged_models'], X_year)
        rf_pred = pd.Series(m['rf_model'].predict(X_year), index=X_year.index)
        ens = 0.6 * bagged + 0.4 * rf_pred
        pred.loc[mask] = ens
    return pred.dropna().groupby(level='date').rank(pct=True)


def make_signal_fn(predictions, returns_df, regime_series,
                   max_sector_weight=MAX_SECTOR_WEIGHT):
    pp = predictions.unstack('symbol')
    def sf(date):
        if date in pp.index:
            preds = pp.loc[date]
        else:
            prior = pp.loc[:date]
            if prior.empty:
                return pd.Series(0.0, index=pp.columns)
            preds = prior.iloc[-1]
        valid = preds.dropna()
        if len(valid) < 2:
            return pd.Series(0.0, index=pp.columns)

        bull = regime_series.loc[:date]
        if bull.empty:
            return pd.Series(0.0, index=pp.columns)
            
        regime_factor = float(bull.iloc[-1]) # Continuous 0.0 to 1.0 (or higher if max_leverage > 1)
        if regime_factor <= 0:
            return pd.Series(0.0, index=pp.columns)

        rw = returns_df.loc[:date].tail(COV_WINDOW)
        w = sector_neutral_top_k(
            valid, rw, k=TOP_K,
            max_position=MAX_POSITION,
            max_sector_weight=max_sector_weight,
            min_sectors=MIN_SECTORS,
        )
        out = pd.Series(0.0, index=pp.columns)
        out.loc[w.index] = w.values * regime_factor
        return out
    return sf


def main():
    print("=" * 75)
    print("PHASE 5 FINAL — Tuned LGBM + bag + RF + composite regime + sector cap")
    print("=" * 75)

    # ---- Load tuning + panel ----
    print("\n[1/5] Loading tuning artifacts and panel...")
    with open('data/models/p4_tuning.pkl', 'rb') as f:
        tuning = pickle.load(f)
    tuned_params = dict(tuning['best_params'])
    tuned_n_est = tuned_params.pop('n_estimators')
    print(f"  Tuned LGBM params (from Optuna): n_est={tuned_n_est}, lr={tuned_params['learning_rate']:.4f}")

    cs_panel = pd.read_parquet('data/processed/features/CS_PANEL.parquet')
    market_context = pd.read_parquet('data/processed/features/MARKET_CONTEXT.parquet')
    panel = assemble_panel_features(cs_panel, market_context)
    labels = build_label_panel(horizon=5, method='pct_rank')
    X, y = align_panel_features_labels(panel, labels, feature_lag=1)
    dates = X.index.get_level_values('date')
    print(f"  Panel: {X.shape}")

    # ---- Walk-forward training (tuned + bagged + RF) ----
    print(f"\n[2/5] Walk-forward training (5-seed bagged tuned LGBM + RF)...")
    backtest_start_year = pd.Timestamp(BACKTEST_START).year
    end_year = dates.max().year
    year_models = {}

    for year in range(backtest_start_year, end_year + 1):
        cutoff = pd.Timestamp(f'{year}-01-01')
        train_mask = dates < cutoff
        if train_mask.sum() < 8000:
            print(f"  {year}: skipped ({train_mask.sum()} samples)")
            continue
        X_tr, y_tr = X[train_mask], y[train_mask]
        print(f"  {year}: training on {train_mask.sum()} samples...", end='', flush=True)

        bagged = train_bagged_with_seeds(X_tr, y_tr, tuned_params, tuned_n_est)
        rf = make_rf().fit(X_tr, y_tr)

        # Evaluate on the upcoming year
        test_mask = (dates >= cutoff) & (dates < pd.Timestamp(f'{year+1}-01-01'))
        oos = None
        if test_mask.sum() > 100:
            X_te, y_te = X[test_mask], y[test_mask]
            preds = predict_bagged(bagged, X_te)
            rf_pred = pd.Series(rf.predict(X_te), index=X_te.index)
            ens = 0.6 * preds + 0.4 * rf_pred
            oos = evaluate_predictions(y_te, ens)
            print(f" OOS IC = {oos['mean_daily_ic']:+.4f} (t={oos['ic_t_stat']:+.2f})")
        else:
            print(" (no OOS year)")

        year_models[year] = {
            'bagged_models': bagged,
            'rf_model': rf,
            'feature_names': X.columns.tolist(),
            'oos_metrics': oos,
        }

    # Save
    Path('data/models').mkdir(parents=True, exist_ok=True)
    with open('data/models/p5_year_models.pkl', 'wb') as f:
        pickle.dump(year_models, f)

    ics = [m['oos_metrics']['mean_daily_ic'] for m in year_models.values()
           if m.get('oos_metrics')]
    if ics:
        print(f"\n  Mean OOS IC: {np.mean(ics):+.4f}, positive years: {sum(1 for i in ics if i > 0)}/{len(ics)}")

    # ---- Predictions ----
    print(f"\n[3/5] Generating predictions...")
    predictions = predict_panel_walk_forward(X, year_models).dropna()
    print(f"  Predictions: {len(predictions)}")

    # ---- Setup backtest ----
    universe = sorted([f.stem for f in Path('data/processed').glob('*.parquet')
                       if f.stem != 'BENCHMARK_INDEX'])
    prices = {s: pd.read_parquet(f'data/processed/{s}.parquet')['Close'] for s in universe}
    volumes_dict = {s: pd.read_parquet(f'data/processed/{s}.parquet')['Volume'] for s in universe}
    
    price_df = pd.DataFrame(prices).dropna(how='all')
    volume_df = pd.DataFrame(volumes_dict).dropna(how='all')
    returns_df = price_df.pct_change()
    
    nifty_df = pd.read_parquet('data/processed/BENCHMARK_INDEX.parquet')
    nifty_close = nifty_df['Close']

    bt_prices = price_df.loc[BACKTEST_START:]
    bt_volumes = volume_df.loc[BACKTEST_START:]
    rebalance_dates = get_rebalance_dates(bt_prices.index, frequency='monthly')
    config = BacktestConfig(initial_capital=1_000_000, slippage_bps=1.0, commission_bps=5.0)

    # Two regime variants
    regime_simple = sma_regime(nifty_close, window=200)
    regime_composite = composite_bullish_signal(nifty_df)
    print(f"\n  Simple regime (SMA200): {regime_simple.mean():.1%} bullish")
    print(f"  Composite continuous regime mean scale: {regime_composite.mean():.2f}")

    # ---- Run variants ----
    print(f"\n[4/5] Running strategy variants...")
    variants = {}

    # P5 with composite regime
    sig_v2 = make_signal_fn(predictions, returns_df, regime_composite)
    variants['P5 (tuned + composite regime)'] = run_backtest(bt_prices, rebalance_dates, sig_v2, config, volumes=bt_volumes)

    # P5 with simple regime (for comparison)
    sig_v1 = make_signal_fn(predictions, returns_df, regime_simple)
    variants['P5 (tuned + simple regime)'] = run_backtest(bt_prices, rebalance_dates, sig_v1, config, volumes=bt_volumes)

    # Equal-weight benchmark
    eq_signal = equal_weight_signal(list(bt_prices.columns))
    variants['Equal-Weight'] = run_backtest(bt_prices, rebalance_dates, eq_signal, config, volumes=bt_volumes)

    for name, r in variants.items():
        m = compute_metrics(r.equity_curve, label=name)
        print_metrics(m)
        if 'P5' in name:
            print(f"  Avg fraction invested: {r.weights.sum(axis=1).mean():.2%}")
            print(f"  Avg turnover/rebalance: {r.turnover.mean():.2%}")

    # ---- Compare to all phases ----
    print("\n" + "=" * 75)
    print("ALL-PHASE COMPARISON")
    print("=" * 75)
    curves = {name: r.equity_curve for name, r in variants.items()}
    for name, fname in [
        ('Phase 1 (per-stock)', 'ml_wf_equity.csv'),
        ('Phase 2 (cross-sectional)', 'p2_equity.csv'),
        ('Phase 3 (bagged + sector)', None),  # find dynamically
        ('Phase 4 (multi-horizon)', 'p4_equity.csv'),
    ]:
        if fname is None:
            matches = list(Path('data/backtest_results').glob('p3_*sector-neutral*equity.csv'))
            if matches:
                try:
                    curves[name] = pd.read_csv(matches[0], index_col=0, parse_dates=True).iloc[:, 0]
                except Exception:
                    pass
        else:
            path = Path(f'data/backtest_results/{fname}')
            if path.exists():
                try:
                    curves[name] = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
                except Exception:
                    pass

    cmp = compare_strategies(curves)
    print(cmp.round(4).to_string())

    # ---- Plots ----
    print("\n[5/5] Generating plots...")
    out_dir = Path('data/backtest_results')
    plot_equity_curves(curves, save_path=str(out_dir / 'p5_equity_curves.png'),
                       title='Phase 5 vs All Predecessors',
                       benchmark_key='Equal-Weight 46')
    p5_eq = variants['P5 (tuned + composite regime)'].equity_curve
    plot_drawdown(p5_eq, save_path=str(out_dir / 'p5_drawdown.png'),
                  title='Drawdown — Phase 5 FINAL')
    plot_monthly_returns(p5_eq, save_path=str(out_dir / 'p5_monthly_returns.png'),
                         title='Monthly Returns — Phase 5 FINAL')
    plot_rolling_sharpe(p5_eq, save_path=str(out_dir / 'p5_rolling_sharpe.png'),
                        title='Rolling 1-Year Sharpe — Phase 5 FINAL')

    # Save artifacts
    p5_eq.to_csv(out_dir / 'p5_equity.csv')
    variants['P5 (tuned + composite regime)'].weights.to_csv(out_dir / 'p5_weights.csv')
    cmp.to_csv(out_dir / 'p5_comparison.csv')

    print(f"\n[OK] Done. Plots and CSVs in {out_dir}")
    print("=" * 75)


if __name__ == '__main__':
    main()
