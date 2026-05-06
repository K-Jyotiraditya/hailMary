"""
Phase 2.1 Backtest — LightGBM+RF ensemble + vol-targeted top-K + regime filter.

Run: PYTHONPATH=. python scripts/p21_backtest.py
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
from ml.cs_model import assemble_panel_features, align_panel_features_labels
from ml.ensemble import ensemble_predict
from portfolio.optimizer import hrp_weights_robust
from portfolio.regime_filter import sma_regime
from portfolio.rebalancer import get_rebalance_dates
from portfolio.vol_targeting import vol_target_with_floor
from backtest.engine import run_backtest, BacktestConfig, equal_weight_signal
from backtest.results import compute_metrics, print_metrics, compare_strategies


TOP_K = 10
COV_WINDOW = 60
MAX_POSITION = 0.18
TARGET_VOL = 0.13       # 13% annualized portfolio vol
VOL_WINDOW = 30
MIN_EXPOSURE = 0.30


def predict_ensemble_panel(panel_features, year_models, rerank: bool = True):
    """Generate ensemble (lgbm + rf) predictions per year, indexed by (date, symbol)."""
    dates = panel_features.index.get_level_values('date')
    years = dates.year

    pred = pd.Series(np.nan, index=panel_features.index)
    for year in sorted(year_models.keys()):
        m = year_models[year]
        mask = (years == year)
        if not mask.any():
            continue
        X_year = panel_features[mask].reindex(columns=m['feature_names'])
        lgbm_p = pd.Series(m['lgbm'].predict(X_year), index=X_year.index)
        rf_p = pd.Series(m['rf'].predict(X_year), index=X_year.index)
        ens = ensemble_predict(lgbm_p, rf_p, w_lgbm=0.6, w_rf=0.4, rerank=rerank)
        pred.loc[mask] = ens
    return pred


def make_p21_signal(predictions, returns_df, regime_series,
                    top_k=TOP_K, cov_window=COV_WINDOW,
                    max_position=MAX_POSITION,
                    target_vol=TARGET_VOL, vol_window=VOL_WINDOW,
                    min_exposure=MIN_EXPOSURE,
                    use_regime=True, use_vol_target=True):
    """Top-K by ensemble + HRP within K + vol-targeted exposure + regime filter."""
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

        # Regime filter (full cash if bearish)
        if use_regime:
            bull = regime_series.loc[:date]
            if bull.empty or not bool(bull.iloc[-1]):
                return pd.Series(0.0, index=pred_pivot.columns)

        top_syms = valid.nlargest(top_k).index.tolist()
        ret_window = returns_df.loc[:date].tail(cov_window)
        ret_window_top = ret_window[top_syms].dropna(how='any')

        if len(ret_window_top) >= 30:
            try:
                hrp_w = hrp_weights_robust(ret_window_top)
            except Exception:
                hrp_w = pd.Series(1.0 / len(top_syms), index=top_syms)
        else:
            hrp_w = pd.Series(1.0 / len(top_syms), index=top_syms)

        hrp_w = hrp_w.clip(upper=max_position)
        if hrp_w.sum() > 0:
            hrp_w = hrp_w / hrp_w.sum()

        # Build full universe weight vector
        out = pd.Series(0.0, index=pred_pivot.columns)
        out.loc[top_syms] = hrp_w.values

        # Vol target the whole vector
        if use_vol_target:
            out = vol_target_with_floor(out, returns_df, target_vol=target_vol,
                                        window=vol_window,
                                        min_exposure=min_exposure,
                                        max_leverage=1.0)
        return out

    return signal_fn


def main():
    print("=" * 75)
    print("PHASE 2.1 BACKTEST — Ensemble + Vol-Target + Regime Filter")
    print("=" * 75)

    print("\n[1/5] Loading models and panel...")
    with open('data/models/p21_year_models.pkl', 'rb') as f:
        year_models = pickle.load(f)

    cs_panel = pd.read_parquet('data/processed/features/CS_PANEL.parquet')
    market_context = pd.read_parquet('data/processed/features/MARKET_CONTEXT.parquet')
    panel = assemble_panel_features(cs_panel, market_context)
    labels = build_label_panel(horizon=5, method='pct_rank')
    X, y = align_panel_features_labels(panel, labels, feature_lag=1)

    print("\n[2/5] Generating ensemble predictions...")
    predictions = predict_ensemble_panel(X, year_models).dropna()
    print(f"  Predictions: {len(predictions)}")

    print("\n[3/5] Loading prices, regime...")
    universe = sorted([f.stem for f in Path('data/processed').glob('*.parquet')
                       if f.stem != 'BENCHMARK_INDEX'])
    prices = {s: pd.read_parquet(f'data/processed/{s}.parquet')['Close'] for s in universe}
    price_df = pd.DataFrame(prices).dropna(how='all')
    returns_df = price_df.pct_change()
    nifty = pd.read_parquet('data/processed/BENCHMARK_INDEX.parquet')['Close']
    regime = sma_regime(nifty, window=200)

    bt_prices = price_df.loc['2019-01-01':]
    rebalance_dates = get_rebalance_dates(bt_prices.index, frequency='monthly')
    config = BacktestConfig(initial_capital=1_000_000, slippage_bps=1.0, commission_bps=5.0)

    # Strategy variants to compare
    print("\n[4/5] Running strategy variants...")
    variants = {}

    # Phase 2.1: Full stack (ensemble + regime + vol target)
    sig_full = make_p21_signal(predictions, returns_df, regime,
                               use_regime=True, use_vol_target=True)
    variants['P2.1 Full (ens+regime+volt)'] = run_backtest(bt_prices, rebalance_dates, sig_full, config)

    # Ensemble + regime, no vol target
    sig_no_vt = make_p21_signal(predictions, returns_df, regime,
                                 use_regime=True, use_vol_target=False)
    variants['P2.1 Ens+regime (no vol-t)'] = run_backtest(bt_prices, rebalance_dates, sig_no_vt, config)

    # Ensemble + vol target, no regime
    sig_no_reg = make_p21_signal(predictions, returns_df, regime,
                                  use_regime=False, use_vol_target=True)
    variants['P2.1 Ens+volt (no regime)'] = run_backtest(bt_prices, rebalance_dates, sig_no_reg, config)

    # Equal-weight 46 baseline
    eq_signal = equal_weight_signal(list(bt_prices.columns))
    variants['Equal-Weight 46'] = run_backtest(bt_prices, rebalance_dates, eq_signal, config)

    # Print metrics
    for name, r in variants.items():
        m = compute_metrics(r.equity_curve, label=name)
        print_metrics(m)
        if 'P2.1' in name:
            print(f"  Avg fraction invested: {r.weights.sum(axis=1).mean():.2%}")
            print(f"  Avg turnover/rebalance: {r.turnover.mean():.2%}")

    # ---- Comparison ----
    print("\n" + "=" * 75)
    print("COMPARISON")
    print("=" * 75)
    cmp = compare_strategies({name: r.equity_curve for name, r in variants.items()})
    print(cmp.to_string())

    # Save
    print("\n[5/5] Saving results...")
    out_dir = Path('data/backtest_results')
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, r in variants.items():
        safe_name = name.replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '').replace(',', '')
        r.equity_curve.to_csv(out_dir / f'p21_{safe_name}.csv')
    cmp.to_csv(out_dir / 'p21_comparison.csv')
    print(f"  Saved to {out_dir}/")
    print("=" * 75)


if __name__ == '__main__':
    main()
