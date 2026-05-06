"""
Phase 2 validation: walk-forward, bootstrap CI, PBO across variants, DSR, cost stress.

Adapts the Phase 1 validation suite to the cross-sectional top-K + regime
strategy. Key strategy variants for PBO: top_k {5,10,15,20}, regime on/off,
cov_window {30, 60, 90}, max_position {0.15, 0.20, 0.25}.

Run: PYTHONPATH=. python scripts/p2_validate.py
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from features.cross_sectional import build_cross_sectional_features
from features.market_context import build_market_context
from labels.rank_label import build_label_panel
from ml.cs_model import assemble_panel_features, align_panel_features_labels
from portfolio.regime_filter import sma_regime
from portfolio.rebalancer import get_rebalance_dates
from backtest.engine import run_backtest, BacktestConfig
from backtest.results import compute_metrics, sharpe_ratio
from validation.walk_forward import (
    walk_forward_analysis, summarize_walk_forward, bootstrap_sharpe_ci,
)
from validation.pbo import compute_pbo, interpret_pbo
from validation.dsr import dsr_from_returns, interpret_dsr

from scripts.p2_backtest import predict_panel_with_walk_forward, make_p2_signal_fn


def setup():
    """Load everything needed for validation."""
    with open('data/models/p2_year_models.pkl', 'rb') as f:
        year_models = pickle.load(f)

    cs_panel = pd.read_parquet('data/processed/features/CS_PANEL.parquet')
    market_context = pd.read_parquet('data/processed/features/MARKET_CONTEXT.parquet')
    panel = assemble_panel_features(cs_panel, market_context)
    labels = build_label_panel(horizon=5, method='pct_rank')
    X, y = align_panel_features_labels(panel, labels, feature_lag=1)

    predictions = predict_panel_with_walk_forward(X, year_models).dropna()

    universe = sorted([f.stem for f in Path('data/processed').glob('*.parquet')
                       if f.stem != 'BENCHMARK_INDEX'])
    prices = {s: pd.read_parquet(f'data/processed/{s}.parquet')['Close'] for s in universe}
    price_df = pd.DataFrame(prices).dropna(how='all')
    returns_df = price_df.pct_change()

    nifty = pd.read_parquet('data/processed/BENCHMARK_INDEX.parquet')['Close']
    regime = sma_regime(nifty, window=200)

    bt_prices = price_df.loc['2019-01-01':]
    rebalance_dates = get_rebalance_dates(bt_prices.index, frequency='monthly')

    return predictions, returns_df, regime, bt_prices, rebalance_dates


def run_variant(predictions, returns_df, regime, bt_prices, rebalance_dates,
                top_k=10, cov_window=60, max_position=0.18,
                use_regime=True, slippage_bps=1.0, commission_bps=5.0):
    """Run one strategy variant and return metrics + return series."""
    sig = make_p2_signal_fn(predictions, returns_df, regime,
                            top_k=top_k, cov_window=cov_window,
                            max_position=max_position, use_regime=use_regime)
    cfg = BacktestConfig(initial_capital=1_000_000,
                         slippage_bps=slippage_bps,
                         commission_bps=commission_bps)
    r = run_backtest(bt_prices, rebalance_dates, sig, cfg)
    m = compute_metrics(r.equity_curve)
    return {
        'sharpe': m['sharpe'],
        'cagr': m['cagr'],
        'max_drawdown': m['max_drawdown'],
        'returns': r.equity_curve.pct_change().dropna(),
    }


def main():
    print("=" * 75)
    print("PHASE 2 VALIDATION SUITE")
    print("=" * 75)

    print("\n[Setup] Loading predictions, prices, regime...")
    predictions, returns_df, regime, bt_prices, rebalance_dates = setup()
    print(f"  Predictions: {len(predictions)}")
    print(f"  Backtest period: {bt_prices.index[0].date()} - {bt_prices.index[-1].date()}")

    # ---- 1. PRODUCTION (regime-filtered) ----
    print("\n[1/5] Production strategy (Top-10 + HRP + regime filter)...")
    prod = run_variant(predictions, returns_df, regime, bt_prices, rebalance_dates,
                       top_k=10, cov_window=60, max_position=0.18, use_regime=True)
    print(f"  Sharpe: {prod['sharpe']:.4f}")
    print(f"  CAGR:   {prod['cagr']:.2%}")
    print(f"  MaxDD:  {prod['max_drawdown']:.2%}")

    # Walk-forward folds
    wf_df = walk_forward_analysis(prod['returns'], n_folds=4)
    wf_summary = summarize_walk_forward(wf_df)
    print(f"\n  Walk-forward folds:")
    print(wf_df[['train_end', 'test_end', 'is_sharpe', 'oos_sharpe', 'degradation']]
          .round(4).to_string(index=False))
    print(f"\n  Mean OOS Sharpe:  {wf_summary['mean_oos_sharpe']:+.4f}")
    print(f"  Std OOS Sharpe:   {wf_summary['std_oos_sharpe']:.4f}")
    print(f"  Mean degradation: {wf_summary['mean_degradation']:+.4f}")

    # Bootstrap CI
    ci = bootstrap_sharpe_ci(prod['returns'], n_bootstrap=3000, block_size=20)
    print(f"\n  Bootstrap Sharpe 95% CI: [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
    print(f"  P(Sharpe <= 0): {ci['p_value']:.4f}")

    # ---- 2. PBO across variants ----
    print("\n[2/5] Generating strategy variants for PBO...")
    grid = list(product(
        [5, 8, 10, 12, 15, 20],          # top_k
        [30, 60, 90],                     # cov_window
        [0.15, 0.20, 0.25],              # max_position
        [True, False],                    # use_regime
    ))
    print(f"  {len(grid)} variants")

    variants_returns = []
    variant_names = []
    for i, (k, cw, mp, ur) in enumerate(grid):
        v = run_variant(predictions, returns_df, regime, bt_prices, rebalance_dates,
                        top_k=k, cov_window=cw, max_position=mp, use_regime=ur)
        variants_returns.append(v['returns'])
        variant_names.append(f"k{k}_cw{cw}_mp{mp}_reg{int(ur)}")
        if (i + 1) % 18 == 0:
            print(f"    [{i+1}/{len(grid)}]")

    returns_panel = pd.concat(variants_returns, axis=1)
    returns_panel.columns = variant_names
    returns_panel = returns_panel.dropna(how='all').fillna(0)

    pbo_result = compute_pbo(returns_panel, S=16)
    print()
    print(interpret_pbo(pbo_result))

    # ---- 3. DSR ----
    print("\n[3/5] Deflated Sharpe Ratio...")
    variant_sharpes = [sharpe_ratio(returns_panel[c].dropna()) for c in returns_panel.columns]
    var_sharpes = float(np.var(variant_sharpes))
    print(f"  Variance of Sharpes: {var_sharpes:.4f}")
    print(f"  Mean Sharpe:         {np.mean(variant_sharpes):.4f}")
    print(f"  Best Sharpe:         {max(variant_sharpes):.4f}")

    dsr = dsr_from_returns(prod['returns'], n_trials=len(grid),
                           variance_of_sharpes=var_sharpes, benchmark_sharpe=0.0)
    print()
    print(interpret_dsr(dsr))

    # ---- 4. COST STRESS ----
    print("\n[4/5] Cost stress test...")
    rows = []
    for slip in [0.5, 1.0, 2.0, 5.0]:
        for comm in [2.5, 5.0, 10.0]:
            v = run_variant(predictions, returns_df, regime, bt_prices, rebalance_dates,
                            slippage_bps=slip, commission_bps=comm)
            rows.append({'slip': slip, 'comm': comm, 'sharpe': v['sharpe'],
                         'cagr': v['cagr'], 'maxdd': v['max_drawdown']})
    stress_df = pd.DataFrame(rows)
    pivot = stress_df.pivot(index='slip', columns='comm', values='sharpe')
    print("\nSharpe by slippage(bps) x commission(bps):")
    print(pivot.round(3).to_string())

    # ---- 5. FINAL CHECKS ----
    print("\n" + "=" * 75)
    print("FINAL VALIDATION REPORT — PHASE 2")
    print("=" * 75)
    checks = {
        'OOS Sharpe > 0.5':              prod['sharpe'] > 0.5,
        'OOS Sharpe > 1.0':              prod['sharpe'] > 1.0,
        'OOS Sharpe CI excludes 0':       ci['ci_lower'] > 0,
        'P(Sharpe <= 0) < 0.05':          ci['p_value'] < 0.05,
        'WF mean OOS Sharpe > 0':         wf_summary['mean_oos_sharpe'] > 0,
        'WF degradation < 30%':           wf_summary['mean_degradation'] < 0.30,
        'PBO < 0.50':                     pbo_result['pbo'] < 0.50,
        'DSR > 0.50':                     (dsr.get('dsr') or 0) > 0.50,
        'DSR > 0.95 (gold)':              (dsr.get('dsr') or 0) > 0.95,
        'Robust to costs':                (stress_df['sharpe'] > 0.5).mean() > 0.75,
        'MaxDD < 25%':                    prod['max_drawdown'] > -0.25,
    }
    for chk, passed in checks.items():
        status = '[PASS]' if passed else '[FAIL]'
        print(f"  {status} {chk}")

    n_passed = sum(checks.values())
    print(f"\n  {n_passed}/{len(checks)} checks passed")
    verdict = ('STRONG GO' if n_passed >= 9
               else 'GO' if n_passed >= 7
               else 'CAUTION' if n_passed >= 5
               else 'NO-GO')
    print(f"\n  VERDICT: {verdict}")

    # ---- Save ----
    out_dir = Path('data/backtest_results')
    out_dir.mkdir(parents=True, exist_ok=True)
    wf_df.to_csv(out_dir / 'p2_walk_forward.csv', index=False)
    returns_panel.to_csv(out_dir / 'p2_variant_returns.csv')
    stress_df.to_csv(out_dir / 'p2_cost_stress.csv', index=False)
    pd.Series({
        'production': {'sharpe': prod['sharpe'], 'cagr': prod['cagr'],
                       'max_drawdown': prod['max_drawdown']},
        'walk_forward': wf_summary,
        'bootstrap_ci': ci,
        'pbo': {k: v for k, v in pbo_result.items() if k != 'logit_distribution'},
        'dsr': dsr,
        'checks': checks,
        'n_passed': n_passed,
        'verdict': verdict,
    }).to_pickle(out_dir / 'p2_validation_summary.pkl')

    print(f"\n[OK] Saved to {out_dir}/")
    print("=" * 75)


if __name__ == '__main__':
    main()
