"""
Phase 4 Optimal Pipeline — combine ALL improvements:
  - Selected features (IC-stable subset)
  - Optuna-tuned LightGBM hyperparameters
  - Multi-horizon ensemble (1d + 5d + 20d)
  - Bagged LGBM + Random Forest blending
  - Sector-neutral top-K + regime filter

Run: PYTHONPATH=. python scripts/p4_optimal.py
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
from portfolio.rebalancer import get_rebalance_dates
from portfolio.sector_neutral import sector_neutral_top_k
from data.sectors import sector_distribution
from backtest.engine import run_backtest, BacktestConfig, equal_weight_signal
from backtest.results import compute_metrics, print_metrics, compare_strategies
from backtest.visualize import (
    plot_equity_curves, plot_drawdown, plot_monthly_returns, plot_rolling_sharpe,
)


# Phase 4 config
TOP_K = 10
COV_WINDOW = 60
MAX_POSITION = 0.18
MAX_SECTOR_WEIGHT = 0.35
MIN_SECTORS = 4
N_BAGGING_SEEDS = 5
HORIZONS = [1, 5, 20]                      # multi-horizon
HORIZON_WEIGHTS = {1: 0.20, 5: 0.55, 20: 0.25}  # 5d emphasized
USE_REGIME = True
BACKTEST_START = '2019-01-01'


def make_lgbm(params: dict, n_estimators: int):
    """Factory returning a fresh LGBM regressor with the supplied tuned params."""
    base = {
        'objective': 'regression', 'metric': 'rmse',
        'verbosity': -1, 'random_state': 42, 'n_jobs': -1,
    }
    return lgb.LGBMRegressor(n_estimators=n_estimators, **{**base, **params})


def make_rf():
    return RandomForestRegressor(
        n_estimators=200, max_depth=12,
        min_samples_split=100, min_samples_leaf=50,
        max_features=0.5, n_jobs=-1, random_state=42,
    )


def train_bagged_lgbm_tuned(X, y, params, n_estimators, n_seeds=N_BAGGING_SEEDS):
    """Train n_seeds copies of LGBM with seed jittering."""
    models = []
    for i in range(n_seeds):
        p = dict(params)
        p['random_state'] = 42 + i
        p['feature_fraction_seed'] = 42 + i
        p['bagging_seed'] = 42 + i
        # Slight jitter for decorrelation
        p['colsample_bytree'] = float(np.clip(
            params.get('colsample_bytree', 0.8) * (0.95 + 0.10 * i / n_seeds), 0.4, 1.0))
        p['subsample'] = float(np.clip(
            params.get('subsample', 0.8) * (0.95 + 0.10 * i / n_seeds), 0.4, 1.0))
        models.append(make_lgbm(p, n_estimators).fit(X, y))
    return models


def main():
    print("=" * 75)
    print("PHASE 4 OPTIMAL — selected features + tuned LGBM + multi-horizon + bag + RF + sector + regime")
    print("=" * 75)

    # ---- Load tuning results ----
    print("\n[1/6] Loading tuning artifacts...")
    with open('data/models/p4_tuning.pkl', 'rb') as f:
        tuning = pickle.load(f)
    best_params = tuning['best_params']
    selected_features = tuning['selected_features']
    n_estimators = best_params.pop('n_estimators')
    print(f"  Tuned LGBM params: {best_params}")
    print(f"  n_estimators: {n_estimators}")
    print(f"  Selected features: {len(selected_features)}")

    # ---- Load panel ----
    print("\n[2/6] Loading feature panel and multi-horizon labels...")
    cs_panel = pd.read_parquet('data/processed/features/CS_PANEL.parquet')
    market_context = pd.read_parquet('data/processed/features/MARKET_CONTEXT.parquet')
    panel = assemble_panel_features(cs_panel, market_context)

    label_panels = {h: build_label_panel(horizon=h, method='pct_rank') for h in HORIZONS}
    print(f"  Built {len(HORIZONS)} label panels for horizons {HORIZONS}")

    # Align features (lagged) once for the master horizon
    X_master, _ = align_panel_features_labels(panel, label_panels[HORIZONS[0]], feature_lag=1)
    X_master = X_master[selected_features]
    dates_master = X_master.index.get_level_values('date')
    print(f"  Master panel (selected features): {X_master.shape}")

    # ---- Walk-forward training: per year, per horizon ----
    print(f"\n[3/6] Walk-forward training (multi-horizon, bagged + RF)...")
    backtest_start_year = pd.Timestamp(BACKTEST_START).year
    end_year = dates_master.max().year
    year_models = {}

    for year in range(backtest_start_year, end_year + 1):
        cutoff = pd.Timestamp(f'{year}-01-01')
        train_mask = dates_master < cutoff
        if train_mask.sum() < 8000:
            print(f"  {year}: skipped ({train_mask.sum()} samples)")
            continue

        X_tr = X_master[train_mask]

        # Train per-horizon models
        horizon_models = {}
        for h in HORIZONS:
            # Get matching y for this horizon
            X_h, y_h = align_panel_features_labels(panel, label_panels[h], feature_lag=1)
            X_h = X_h[selected_features]
            dates_h = X_h.index.get_level_values('date')
            mask_h = dates_h < cutoff
            X_h_tr, y_h_tr = X_h[mask_h], y_h[mask_h]
            if len(X_h_tr) < 5000:
                continue

            bagged = train_bagged_lgbm_tuned(X_h_tr, y_h_tr, best_params, n_estimators)
            rf = make_rf().fit(X_h_tr, y_h_tr)
            horizon_models[h] = {'bagged_models': bagged, 'rf_model': rf}

        # Quick OOS evaluation on the upcoming year (5d horizon for primary IC)
        primary_h = 5
        X_5h, y_5h = align_panel_features_labels(panel, label_panels[primary_h], feature_lag=1)
        X_5h = X_5h[selected_features]
        dates_5h = X_5h.index.get_level_values('date')
        test_mask = (dates_5h >= cutoff) & (dates_5h < pd.Timestamp(f'{year+1}-01-01'))
        oos_metrics = None
        if test_mask.sum() > 100:
            X_te, y_te = X_5h[test_mask], y_5h[test_mask]
            preds = predict_bagged(horizon_models[primary_h]['bagged_models'], X_te)
            rf_pred = horizon_models[primary_h]['rf_model'].predict(X_te)
            ens = 0.6 * preds + 0.4 * pd.Series(rf_pred, index=preds.index)
            oos_metrics = evaluate_predictions(y_te, ens)
            print(f"  {year}: {len(horizon_models)}/{len(HORIZONS)} horizons trained | "
                  f"OOS 5d IC = {oos_metrics['mean_daily_ic']:+.4f} (t={oos_metrics['ic_t_stat']:+.2f})")
        else:
            print(f"  {year}: trained {len(horizon_models)}/{len(HORIZONS)} horizons (no OOS year)")

        year_models[year] = {
            'horizon_models': horizon_models,
            'feature_names': selected_features,
            'oos_metrics': oos_metrics,
        }

    # Save
    Path('data/models').mkdir(parents=True, exist_ok=True)
    with open('data/models/p4_year_models.pkl', 'wb') as f:
        pickle.dump(year_models, f)

    ics = [m['oos_metrics']['mean_daily_ic'] for m in year_models.values()
           if m.get('oos_metrics')]
    if ics:
        print(f"\n  Mean OOS IC across years: {np.mean(ics):+.4f}")
        print(f"  Years with positive IC:   {sum(1 for ic in ics if ic > 0)}/{len(ics)}")

    # ---- Predictions: multi-horizon ensemble per (date, symbol) ----
    print(f"\n[4/6] Generating multi-horizon ensemble predictions...")
    predictions = pd.Series(np.nan, index=X_master.index)
    for year, m in year_models.items():
        mask = (dates_master.year == year)
        if not mask.any():
            continue
        X_year = X_master[mask]
        ensemble = pd.Series(0.0, index=X_year.index)
        for h, hm in m['horizon_models'].items():
            bagged = predict_bagged(hm['bagged_models'], X_year)
            rf_pred = pd.Series(hm['rf_model'].predict(X_year), index=X_year.index)
            blend = 0.6 * bagged + 0.4 * rf_pred
            ensemble = ensemble + HORIZON_WEIGHTS[h] * blend
        predictions.loc[mask] = ensemble.values

    predictions = predictions.dropna().groupby(level='date').rank(pct=True)
    print(f"  Predictions: {len(predictions)}")

    # ---- Backtest ----
    print(f"\n[5/6] Backtesting...")
    universe = sorted([f.stem for f in Path('data/processed').glob('*.parquet')
                       if f.stem != 'BENCHMARK_INDEX'])
    prices = {s: pd.read_parquet(f'data/processed/{s}.parquet')['Close'] for s in universe}
    price_df = pd.DataFrame(prices).dropna(how='all')
    returns_df = price_df.pct_change()
    nifty = pd.read_parquet('data/processed/BENCHMARK_INDEX.parquet')['Close']
    regime = sma_regime(nifty, window=200)

    bt_prices = price_df.loc[BACKTEST_START:]
    rebalance_dates = get_rebalance_dates(bt_prices.index, frequency='monthly')
    config = BacktestConfig(initial_capital=1_000_000, slippage_bps=1.0, commission_bps=5.0)

    pred_pivot = predictions.unstack('symbol')

    def signal_fn(date):
        if date in pred_pivot.index:
            preds = pred_pivot.loc[date]
        else:
            prior = pred_pivot.loc[:date]
            if prior.empty:
                return pd.Series(0.0, index=pred_pivot.columns)
            preds = prior.iloc[-1]

        valid = preds.dropna()
        if len(valid) < TOP_K:
            return pd.Series(0.0, index=pred_pivot.columns)

        if USE_REGIME:
            bull = regime.loc[:date]
            if bull.empty or not bool(bull.iloc[-1]):
                return pd.Series(0.0, index=pred_pivot.columns)

        ret_window = returns_df.loc[:date].tail(COV_WINDOW)
        weights = sector_neutral_top_k(
            valid, ret_window, k=TOP_K,
            max_position=MAX_POSITION,
            max_sector_weight=MAX_SECTOR_WEIGHT,
            min_sectors=MIN_SECTORS,
        )
        out = pd.Series(0.0, index=pred_pivot.columns)
        out.loc[weights.index] = weights.values
        return out

    p4_results = run_backtest(bt_prices, rebalance_dates, signal_fn, config)
    p4_metrics = compute_metrics(p4_results.equity_curve, label='Phase 4 OPTIMAL')
    print_metrics(p4_metrics)

    avg_inv = p4_results.weights.sum(axis=1).mean()
    print(f"  Avg fraction invested: {avg_inv:.2%}")
    print(f"  Avg turnover/rebalance: {p4_results.turnover.mean():.2%}")

    # Equal-weight baseline
    eq_signal = equal_weight_signal(list(bt_prices.columns))
    eq_results = run_backtest(bt_prices, rebalance_dates, eq_signal, config)
    eq_metrics = compute_metrics(eq_results.equity_curve, label='Equal-Weight 46')
    print_metrics(eq_metrics)

    # ---- Compare to all phases ----
    print("\n" + "=" * 75)
    print("ALL-PHASE COMPARISON")
    print("=" * 75)
    curves = {'Phase 4 OPTIMAL': p4_results.equity_curve}

    for name, fname in [
        ('Phase 1 (per-stock)', 'ml_wf_equity.csv'),
        ('Phase 2 (cross-sectional)', 'p2_equity.csv'),
        ('Phase 3 (bagged + sector-neutral)', 'p3_P3_sector-neutral_+_regime_equity.csv'),
        ('Equal-Weight 46', 'eq46_equity.csv'),
    ]:
        path = Path(f'data/backtest_results/{fname}')
        if path.exists():
            try:
                series = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
                curves[name] = series
            except Exception:
                pass

    # If P3 csv name differs, find it
    p3_alt = list(Path('data/backtest_results').glob('p3_*sector-neutral*equity.csv'))
    if 'Phase 3 (bagged + sector-neutral)' not in curves and p3_alt:
        try:
            curves['Phase 3 (bagged + sector-neutral)'] = pd.read_csv(p3_alt[0], index_col=0, parse_dates=True).iloc[:, 0]
        except Exception:
            pass

    cmp = compare_strategies(curves)
    print(cmp.round(4).to_string())

    # ---- Plots ----
    print("\n[6/6] Generating plots...")
    out_dir = Path('data/backtest_results')
    plot_equity_curves(curves, save_path=str(out_dir / 'p4_equity_curves.png'),
                       title='Phase 4 vs All Predecessors',
                       benchmark_key='Equal-Weight 46')
    plot_drawdown(p4_results.equity_curve,
                  save_path=str(out_dir / 'p4_drawdown.png'),
                  title='Drawdown — Phase 4 OPTIMAL')
    plot_monthly_returns(p4_results.equity_curve,
                         save_path=str(out_dir / 'p4_monthly_returns.png'),
                         title='Monthly Returns — Phase 4 OPTIMAL')
    plot_rolling_sharpe(p4_results.equity_curve,
                        save_path=str(out_dir / 'p4_rolling_sharpe.png'),
                        title='Rolling 1-Year Sharpe — Phase 4 OPTIMAL')

    # Save artifacts
    p4_results.equity_curve.to_csv(out_dir / 'p4_equity.csv')
    p4_results.weights.to_csv(out_dir / 'p4_weights.csv')
    cmp.to_csv(out_dir / 'p4_comparison.csv')

    print("\n[OK] Done. Plots and CSVs in", out_dir)
    print("=" * 75)


if __name__ == '__main__':
    main()
