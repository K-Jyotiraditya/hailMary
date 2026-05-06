"""
Phase 3 Pipeline — multi-seed bagged ensemble + sector-neutral top-K.

End-to-end: train walk-forward bagged models, run backtest with sector
neutrality, validate and visualize.

Run: PYTHONPATH=. python scripts/p3_pipeline.py
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from features.cross_sectional import build_cross_sectional_features
from features.market_context import build_market_context
from labels.rank_label import build_label_panel
from ml.cs_model import (
    assemble_panel_features,
    align_panel_features_labels,
    evaluate_predictions,
)
from ml.bagging import train_bagged_lgbm, predict_bagged
from ml.rf_model import train_rf
from ml.ensemble import ensemble_predict
from portfolio.regime_filter import sma_regime
from portfolio.rebalancer import get_rebalance_dates
from portfolio.sector_neutral import sector_neutral_top_k
from data.sectors import sector_distribution
from backtest.engine import run_backtest, BacktestConfig, equal_weight_signal
from backtest.results import compute_metrics, print_metrics, compare_strategies
from backtest.visualize import (
    plot_equity_curves, plot_drawdown, plot_monthly_returns, plot_rolling_sharpe,
)

TOP_K = 10
COV_WINDOW = 60
MAX_POSITION = 0.18
MAX_SECTOR_WEIGHT = 0.35
MIN_SECTORS = 4
N_BAGGING_SEEDS = 5
USE_REGIME = True
BACKTEST_START = '2019-01-01'


def predict_panel_walk_forward(panel_features, year_models, rerank: bool = True):
    """
    Predict per-year using each year's bagged LGBM ensemble + RF, blended.
    Indexed by (date, symbol).
    """
    dates = panel_features.index.get_level_values('date')
    years = dates.year
    pred = pd.Series(np.nan, index=panel_features.index)
    for year in sorted(year_models.keys()):
        m = year_models[year]
        mask = (years == year)
        if not mask.any():
            continue
        X_year = panel_features[mask].reindex(columns=m['feature_names'])
        # Bagged LGBM
        bag = m['bagged_models']
        bagged = predict_bagged(bag, X_year)
        # RF
        rf = m.get('rf_model')
        if rf is not None:
            rf_pred = pd.Series(rf.predict(X_year), index=X_year.index)
            ens = 0.6 * bagged + 0.4 * rf_pred
        else:
            ens = bagged
        pred.loc[mask] = ens
    if rerank:
        pred = pred.groupby(level='date').rank(pct=True)
    return pred


def make_p3_signal(predictions, returns_df, regime_series,
                   top_k=TOP_K, cov_window=COV_WINDOW,
                   max_position=MAX_POSITION,
                   max_sector_weight=MAX_SECTOR_WEIGHT,
                   min_sectors=MIN_SECTORS,
                   use_regime=USE_REGIME):
    """Sector-neutral top-K with regime filter."""
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
        if len(valid) < top_k:
            return pd.Series(0.0, index=pred_pivot.columns)

        if use_regime:
            bull = regime_series.loc[:date]
            if bull.empty or not bool(bull.iloc[-1]):
                return pd.Series(0.0, index=pred_pivot.columns)

        ret_window = returns_df.loc[:date].tail(cov_window)
        weights = sector_neutral_top_k(
            valid, ret_window, k=top_k,
            max_position=max_position,
            max_sector_weight=max_sector_weight,
            min_sectors=min_sectors,
        )
        out = pd.Series(0.0, index=pred_pivot.columns)
        out.loc[weights.index] = weights.values
        return out

    return signal_fn


def main():
    print("=" * 75)
    print("PHASE 3 PIPELINE — Bagged ensemble + Sector-neutral Top-K + Regime filter")
    print("=" * 75)

    # ---- Load panel and labels ----
    print("\n[1/6] Loading panel features and labels...")
    cs_panel = pd.read_parquet('data/processed/features/CS_PANEL.parquet')
    market_context = pd.read_parquet('data/processed/features/MARKET_CONTEXT.parquet')
    panel = assemble_panel_features(cs_panel, market_context)
    labels = build_label_panel(horizon=5, method='pct_rank')
    X, y = align_panel_features_labels(panel, labels, feature_lag=1)
    dates = X.index.get_level_values('date')
    print(f"  Panel: {X.shape}")

    # ---- Walk-forward bagged training ----
    print(f"\n[2/6] Training {N_BAGGING_SEEDS}-seed bagged LGBM per year...")
    backtest_start_year = pd.Timestamp(BACKTEST_START).year
    end_year = dates.max().year
    year_models = {}

    for year in range(backtest_start_year, end_year + 1):
        cutoff = pd.Timestamp(f'{year}-01-01')
        train_mask = dates < cutoff
        if train_mask.sum() < 10000:
            print(f"  {year}: skipped ({train_mask.sum()} samples)")
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        print(f"  {year}: bagged LGBM ({N_BAGGING_SEEDS} seeds) + RF on {train_mask.sum()} samples...",
              end='', flush=True)
        bagged_models = train_bagged_lgbm(X_tr, y_tr, n_seeds=N_BAGGING_SEEDS)
        rf_model = train_rf(X_tr, y_tr)

        # Evaluate ensemble (bagged LGBM + RF)
        test_mask = (dates >= cutoff) & (dates < pd.Timestamp(f'{year+1}-01-01'))
        oos = None
        if test_mask.sum() > 100:
            X_te, y_te = X[test_mask], y[test_mask]
            bagged_pred = predict_bagged(bagged_models, X_te)
            rf_pred = pd.Series(rf_model.predict(X_te), index=X_te.index)
            ens = 0.6 * bagged_pred + 0.4 * rf_pred
            oos = evaluate_predictions(y_te, ens)
            print(f" OOS daily IC = {oos['mean_daily_ic']:+.4f} "
                  f"(t={oos['ic_t_stat']:+.2f})")
        else:
            print(" (no OOS year)")

        year_models[year] = {
            'bagged_models': bagged_models,
            'rf_model': rf_model,
            'feature_names': X.columns.tolist(),
            'oos_metrics': oos,
            'train_size': train_mask.sum(),
        }

    # ---- Save models ----
    out_dir = Path('data/models')
    out_dir.mkdir(parents=True, exist_ok=True)
    save_payload = {
        year: {
            'bagged_models': m['bagged_models'],
            'rf_model': m.get('rf_model'),
            'feature_names': m['feature_names'],
            **{k: v for k, v in m.items() if k not in ('bagged_models', 'rf_model', 'feature_names')}
        }
        for year, m in year_models.items()
    }
    with open(out_dir / 'p3_year_models.pkl', 'wb') as f:
        pickle.dump(save_payload, f)
    print(f"\n  Saved {len(year_models)} bagged year-models")

    # Aggregate IC
    ics = [m['oos_metrics']['mean_daily_ic'] for m in year_models.values()
           if m.get('oos_metrics')]
    if ics:
        print(f"  Mean OOS IC across years: {np.mean(ics):+.4f} +- {np.std(ics):.4f}")
        print(f"  Years with positive IC:   {sum(1 for ic in ics if ic > 0)}/{len(ics)}")

    # ---- Backtest ----
    print(f"\n[3/6] Generating predictions and running backtest...")
    predictions = predict_panel_walk_forward(X, year_models, rerank=True).dropna()

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

    # Variants
    variants = {}

    # P3 production: sector-neutral + regime filter
    sig = make_p3_signal(predictions, returns_df, regime,
                         max_sector_weight=MAX_SECTOR_WEIGHT, use_regime=True)
    variants['P3 (sector-neutral + regime)'] = run_backtest(bt_prices, rebalance_dates, sig, config)

    # P3 without sector neutrality (concentration allowed) — for comparison
    sig_no_sec = make_p3_signal(predictions, returns_df, regime,
                                max_sector_weight=1.0, use_regime=True)
    variants['P3 (no sector cap)'] = run_backtest(bt_prices, rebalance_dates, sig_no_sec, config)

    # Equal-weight 46 baseline
    eq_signal = equal_weight_signal(list(bt_prices.columns))
    variants['Equal-Weight 46'] = run_backtest(bt_prices, rebalance_dates, eq_signal, config)

    # Print metrics
    print()
    for name, r in variants.items():
        m = compute_metrics(r.equity_curve, label=name)
        print_metrics(m)
        if 'P3' in name:
            avg_inv = r.weights.sum(axis=1).mean()
            print(f"  Avg fraction invested:  {avg_inv:.2%}")
            print(f"  Avg turnover/rebalance: {r.turnover.mean():.2%}")

            # Sector concentration analysis on the most recent rebalance
            if not r.weights.empty:
                last_w = r.weights.iloc[-1]
                last_w = last_w[last_w > 0.001]
                if not last_w.empty:
                    sec_dist = sector_distribution(last_w)
                    print(f"  Latest sector distribution:")
                    for sec, w in sorted(sec_dist.items(), key=lambda x: -x[1]):
                        print(f"      {sec:12} {w:.1%}")

    # ---- Comparison ----
    print("\n" + "=" * 75)
    print("COMPARISON")
    print("=" * 75)
    cmp = compare_strategies({name: r.equity_curve for name, r in variants.items()})
    print(cmp.to_string())

    # ---- Visualizations ----
    print(f"\n[4/6] Generating plots...")
    out_plots = Path('data/backtest_results')
    out_plots.mkdir(parents=True, exist_ok=True)
    plot_equity_curves(
        {name: r.equity_curve for name, r in variants.items()},
        save_path=str(out_plots / 'p3_equity_curves.png'),
        title='Phase 3 vs Baselines',
        benchmark_key='Equal-Weight 46',
    )
    primary_eq = variants['P3 (sector-neutral + regime)'].equity_curve
    plot_drawdown(primary_eq, save_path=str(out_plots / 'p3_drawdown.png'),
                  title='Drawdown — Phase 3 (sector-neutral + regime)')
    plot_monthly_returns(primary_eq, save_path=str(out_plots / 'p3_monthly_returns.png'),
                          title='Monthly Returns — Phase 3')
    plot_rolling_sharpe(primary_eq, save_path=str(out_plots / 'p3_rolling_sharpe.png'),
                         title='Rolling 1-Year Sharpe — Phase 3')

    # ---- Save artifacts ----
    print(f"\n[5/6] Saving artifacts...")
    for name, r in variants.items():
        safe = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '')
        r.equity_curve.to_csv(out_plots / f'p3_{safe}_equity.csv')
    cmp.to_csv(out_plots / 'p3_comparison.csv')

    print(f"\n[6/6] Done. Artifacts in {out_plots}/")
    print("=" * 75)


if __name__ == '__main__':
    main()
