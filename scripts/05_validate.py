"""
Full validation suite for the ML strategy.

Runs:
  1. Walk-forward analysis with bootstrap CI on the production strategy.
  2. PBO across many strategy variants (different threshold/cov_window/mode).
  3. Deflated Sharpe Ratio adjusted for n_trials.
  4. Robustness sweeps: slippage/commission, threshold, rebalance frequency.
  5. Final go/no-go report.

Run: PYTHONPATH=. python scripts/05_validate.py
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from labels.triple_barrier import triple_barrier_labels
from ml.preprocessor import align_features_labels
from ml.lgbm_model import make_lgbm_model
from portfolio.optimizer import hrp_weights_robust
from portfolio.position_sizer import compute_target_weights
from portfolio.rebalancer import get_rebalance_dates
from backtest.engine import run_backtest, BacktestConfig
from backtest.results import compute_metrics, sharpe_ratio
from validation.walk_forward import (
    walk_forward_analysis, summarize_walk_forward, bootstrap_sharpe_ci,
)
from validation.pbo import compute_pbo, interpret_pbo
from validation.dsr import dsr_from_returns, interpret_dsr
from validation.robustness import sweep_parameter, cost_stress_test


def load_universe_and_models():
    """Load the tradeable universe and the year-by-year walk-forward models."""
    universe_df = pd.read_csv('data/models/tradeable_universe.csv')
    tradeable = universe_df['symbol'].tolist()

    with open('data/backtest_results/year_models.pkl', 'rb') as f:
        year_models = pickle.load(f)

    features = {s: pd.read_parquet(f'data/processed/features/{s}_ml.parquet') for s in tradeable}
    prices = {s: pd.read_parquet(f'data/processed/{s}.parquet')['Close'] for s in tradeable}
    return tradeable, year_models, features, prices


def build_signal_fn(tradeable, year_models, features, prices,
                    threshold=0.50, cov_window=60, mode='binary',
                    max_position=0.40):
    """Build a signal function with given parameters."""
    returns = pd.DataFrame({s: prices[s].pct_change() for s in tradeable})

    def signal_fn(date):
        year = date.year if date.year in year_models else max([y for y in year_models if y <= date.year], default=None)
        if year is None:
            return pd.Series(0.0, index=tradeable)
        active = year_models.get(year, {})
        if not active:
            return pd.Series(0.0, index=tradeable)

        probs = pd.Series(dtype=float)
        for s in tradeable:
            if s not in active:
                continue
            f_lag = features[s].shift(1).loc[:date].dropna()
            if f_lag.empty:
                continue
            p = active[s]['model'].predict_proba(
                f_lag.iloc[[-1]][active[s]['feature_names']]
            )[0, 1]
            probs[s] = p

        if probs.empty:
            return pd.Series(0.0, index=tradeable)
        rw = returns.loc[:date].tail(cov_window).dropna(how='any')
        hrp_w = (hrp_weights_robust(rw[tradeable]) if len(rw) >= cov_window // 2
                 else pd.Series(1.0 / len(tradeable), index=tradeable))
        return compute_target_weights(probs, hrp_w, threshold=threshold,
                                       mode=mode, max_position=max_position)
    return signal_fn


def quick_backtest(bt_prices, rebalance_dates, signal_fn,
                   slippage_bps=1.0, commission_bps=5.0) -> dict:
    """Run a backtest and return only the headline metrics."""
    cfg = BacktestConfig(initial_capital=1_000_000,
                         slippage_bps=slippage_bps,
                         commission_bps=commission_bps)
    r = run_backtest(bt_prices, rebalance_dates, signal_fn, cfg)
    m = compute_metrics(r.equity_curve)
    return {
        'sharpe': m['sharpe'],
        'cagr': m['cagr'],
        'max_drawdown': m['max_drawdown'],
        'returns': r.equity_curve.pct_change().dropna(),
        'equity': r.equity_curve,
    }


def main():
    print("=" * 75)
    print("STEP 5: VALIDATION & ROBUSTNESS")
    print("=" * 75)

    # ====== SETUP ======
    print("\n[Setup] Loading universe and models...")
    tradeable, year_models, features, prices = load_universe_and_models()
    price_df = pd.DataFrame({s: prices[s] for s in tradeable}).dropna()
    bt_prices = price_df.loc['2019-01-01':]
    rebalance_dates = get_rebalance_dates(bt_prices.index, frequency='monthly')
    print(f"  Universe: {tradeable}")
    print(f"  Backtest period: {bt_prices.index[0].date()} to {bt_prices.index[-1].date()}")
    print(f"  {len(rebalance_dates)} rebalance dates")

    # ====== 1. PRODUCTION STRATEGY METRICS + BOOTSTRAP CI ======
    print("\n[1/5] Production strategy: walk-forward + bootstrap CI...")
    prod_kwargs = dict(threshold=0.50, cov_window=60, mode='binary', max_position=0.40)
    prod_signal = build_signal_fn(tradeable, year_models, features, prices, **prod_kwargs)
    prod_result = quick_backtest(bt_prices, rebalance_dates, prod_signal)
    print(f"  Production Sharpe: {prod_result['sharpe']:.4f}")
    print(f"  Production CAGR:   {prod_result['cagr']:.2%}")
    print(f"  Production MaxDD:  {prod_result['max_drawdown']:.2%}")

    # Walk-forward folds on the equity curve
    wf_df = walk_forward_analysis(prod_result['returns'], n_folds=4)
    wf_summary = summarize_walk_forward(wf_df)
    print(f"\n  Walk-forward folds:")
    print(wf_df[['train_end', 'test_end', 'is_sharpe', 'oos_sharpe', 'degradation']]
          .round(4).to_string(index=False))
    print(f"\n  Mean OOS Sharpe:   {wf_summary['mean_oos_sharpe']:.4f}")
    print(f"  Std OOS Sharpe:    {wf_summary['std_oos_sharpe']:.4f}")
    print(f"  Mean degradation:  {wf_summary['mean_degradation']:.4f}")
    print(f"  Pass min OOS:      {wf_summary['pass_min_oos_sharpe']}")
    print(f"  Pass degradation:  {wf_summary['pass_degradation']}")
    print(f"  Pass consistency:  {wf_summary['pass_consistency']}")

    # Bootstrap CI on Sharpe
    ci = bootstrap_sharpe_ci(prod_result['returns'], n_bootstrap=3000,
                             confidence=0.95, block_size=20)
    print(f"\n  Bootstrap Sharpe 95% CI: [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
    print(f"  P(Sharpe <= 0): {ci['p_value']:.4f}")

    # ====== 2. PBO ACROSS STRATEGY VARIANTS ======
    print("\n[2/5] Building strategy variants for PBO...")
    variants = []
    variant_names = []
    grid = [
        {'threshold': t, 'cov_window': cw, 'mode': m, 'max_position': mp}
        for t in [0.49, 0.50, 0.51, 0.52, 0.53, 0.54]
        for cw in [30, 60, 90]
        for m in ['binary', 'scaled']
        for mp in [0.30, 0.40]
    ]
    print(f"  Generating {len(grid)} variants...")
    for i, params in enumerate(grid):
        sig = build_signal_fn(tradeable, year_models, features, prices, **params)
        r = quick_backtest(bt_prices, rebalance_dates, sig)
        variants.append(r['returns'])
        variant_names.append(f"t{params['threshold']}_cw{params['cov_window']}_{params['mode']}_mp{params['max_position']}")
        if (i + 1) % 12 == 0:
            print(f"    [{i+1}/{len(grid)}]")

    returns_panel = pd.concat(variants, axis=1)
    returns_panel.columns = variant_names
    returns_panel = returns_panel.dropna(how='all').fillna(0)
    print(f"  Returns panel: {returns_panel.shape}")

    pbo_result = compute_pbo(returns_panel, S=16)
    print()
    print(interpret_pbo(pbo_result))

    # ====== 3. DEFLATED SHARPE RATIO ======
    print("\n[3/5] Deflated Sharpe Ratio (adjusts for n_trials selection bias)...")
    # Variance of Sharpes across our strategy variants
    variant_sharpes = [
        sharpe_ratio(returns_panel[col].dropna())
        for col in returns_panel.columns
    ]
    var_sharpes = float(np.var(variant_sharpes))
    print(f"  Variance of Sharpes across variants: {var_sharpes:.4f}")
    print(f"  Mean Sharpe across variants:         {np.mean(variant_sharpes):.4f}")
    print(f"  Best Sharpe:                         {max(variant_sharpes):.4f}")

    dsr_prod = dsr_from_returns(
        prod_result['returns'],
        n_trials=len(grid),
        variance_of_sharpes=var_sharpes,
        benchmark_sharpe=0.0,
    )
    print()
    print(interpret_dsr(dsr_prod))

    # ====== 4. ROBUSTNESS SWEEPS ======
    print("\n[4/5] Cost stress test (slippage x commission)...")

    def stress_runner(slippage_bps=1.0, commission_bps=5.0, **kwargs):
        # Fixed strategy params
        sig = build_signal_fn(tradeable, year_models, features, prices,
                              threshold=0.50, cov_window=60, mode='binary', max_position=0.40)
        return quick_backtest(bt_prices, rebalance_dates, sig,
                              slippage_bps=slippage_bps, commission_bps=commission_bps)

    stress_df = cost_stress_test(
        stress_runner, base_kwargs={},
        slippage_grid=(0.5, 1.0, 2.0, 5.0),
        commission_grid=(2.5, 5.0, 10.0),
    )
    print("\nCost stress (Sharpe by slippage x commission):")
    pivot = stress_df.pivot(index='slippage_bps', columns='commission_bps', values='sharpe')
    print(pivot.round(3).to_string())

    # ====== 5. FINAL GO/NO-GO ======
    print("\n" + "=" * 75)
    print("FINAL VALIDATION REPORT")
    print("=" * 75)

    checks = {
        'OOS Sharpe > 0.3':        prod_result['sharpe'] > 0.3,
        'OOS Sharpe CI excludes 0': ci['ci_lower'] > 0,
        'P(Sharpe <= 0) < 0.10':   ci['p_value'] < 0.10,
        'WF mean OOS Sharpe > 0':  wf_summary['mean_oos_sharpe'] > 0,
        'WF degradation < 30%':    wf_summary['mean_degradation'] < 0.30,
        'PBO < 0.50':              pbo_result['pbo'] < 0.50,
        'DSR > 0.50':              (dsr_prod.get('dsr') or 0) > 0.50,
        'Robust to costs':         (stress_df['sharpe'] > 0).mean() > 0.75,
    }
    for check, passed in checks.items():
        status = '[PASS]' if passed else '[FAIL]'
        print(f"  {status} {check}")

    n_passed = sum(checks.values())
    print(f"\n  {n_passed}/{len(checks)} checks passed")

    verdict = ('GO — strategy passes minimum validation' if n_passed >= 6
               else 'CAUTION — partial validation, needs improvement'
               if n_passed >= 4
               else 'NO-GO — strategy fails validation')
    print(f"\n  VERDICT: {verdict}")

    # ====== SAVE ARTIFACTS ======
    out_dir = Path('data/backtest_results')
    out_dir.mkdir(parents=True, exist_ok=True)

    wf_df.to_csv(out_dir / 'walk_forward.csv', index=False)
    returns_panel.to_csv(out_dir / 'variant_returns.csv')
    stress_df.to_csv(out_dir / 'cost_stress.csv', index=False)

    summary_payload = {
        'production_metrics': {
            'sharpe': prod_result['sharpe'],
            'cagr': prod_result['cagr'],
            'max_drawdown': prod_result['max_drawdown'],
        },
        'walk_forward': wf_summary,
        'bootstrap_ci': ci,
        'pbo': {k: v for k, v in pbo_result.items() if k != 'logit_distribution'},
        'dsr': dsr_prod,
        'checks': checks,
        'n_passed': n_passed,
        'verdict': verdict,
    }
    pd.Series(summary_payload).to_pickle(out_dir / 'validation_summary.pkl')

    print(f"\n[OK] Validation artifacts saved to {out_dir}/")
    print("=" * 75)


if __name__ == '__main__':
    main()
