"""
Phase 2 Backtest: cross-sectional top-K with regime filter + HRP.

Pipeline:
  1. Load year-models from p2_train.py
  2. Pre-compute predicted ranks across (date, symbol) using year-appropriate model
  3. At each rebalance:
     - Read predictions for that date
     - Apply regime filter (NIFTY 50 above 200-DMA)
     - Pick top-K by prediction
     - HRP-weight within top-K
  4. Run backtest, compare against:
     - Phase 1 ML strategy (6-stock per-stock models)
     - Equal-weight 46
     - NIFTY 50 buy & hold

Run: PYTHONPATH=. python scripts/p2_backtest.py
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
from portfolio.optimizer import hrp_weights_robust
from portfolio.regime_filter import sma_regime
from portfolio.rebalancer import get_rebalance_dates
from backtest.engine import run_backtest, BacktestConfig, equal_weight_signal
from backtest.results import compute_metrics, print_metrics, compare_strategies


# Strategy params
TOP_K = 10
COV_WINDOW = 60
MAX_POSITION = 0.18
USE_REGIME_FILTER = True
REGIME_WINDOW = 200
BEAR_EXPOSURE = 0.0


def predict_panel_with_walk_forward(panel_features, year_models):
    """
    For each (date, symbol) in panel_features, use the model trained on data
    BEFORE that date's year to generate predictions.

    Returns: Series indexed by (date, symbol) with predicted rank.
    """
    dates = panel_features.index.get_level_values('date')
    years = dates.year

    pred = pd.Series(np.nan, index=panel_features.index)
    for year in sorted(year_models.keys()):
        m = year_models[year]
        mask = (years == year)
        if not mask.any():
            continue
        X_year = panel_features[mask]
        feat_names = m['feature_names']
        # Use only the columns the model was trained on (in case feature set drifted)
        X_year = X_year.reindex(columns=feat_names)
        pred.loc[mask] = m['model'].predict(X_year)
    return pred


def make_p2_signal_fn(predictions, returns_df, regime_series,
                     top_k=TOP_K, cov_window=COV_WINDOW,
                     max_position=MAX_POSITION, use_regime=USE_REGIME_FILTER,
                     bear_exposure=BEAR_EXPOSURE):
    """Build the signal function for Phase 2."""

    # predictions is a Series indexed by (date, symbol)
    pred_pivot = predictions.unstack('symbol')  # date x symbol matrix

    def signal_fn(date):
        # Find latest prediction date <= signal date
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

        # Regime filter
        if use_regime:
            bull = regime_series.loc[:date]
            if bull.empty or not bool(bull.iloc[-1]):
                # Bearish regime: full cash (or partial via bear_exposure)
                if bear_exposure == 0:
                    return pd.Series(0.0, index=pred_pivot.columns)

        # Top-K
        top_syms = valid.nlargest(top_k).index.tolist()

        # HRP weighting within top K
        ret_window = returns_df.loc[:date].tail(cov_window)
        ret_window_top = ret_window[top_syms].dropna(how='any')
        if len(ret_window_top) >= 30:
            try:
                hrp_w = hrp_weights_robust(ret_window_top)
            except Exception:
                hrp_w = pd.Series(1.0 / len(top_syms), index=top_syms)
        else:
            hrp_w = pd.Series(1.0 / len(top_syms), index=top_syms)

        # Apply position cap, renormalize
        hrp_w = hrp_w.clip(upper=max_position)
        if hrp_w.sum() > 0:
            hrp_w = hrp_w / hrp_w.sum()

        # If bearish but partial exposure, scale down
        if use_regime:
            bull = regime_series.loc[:date]
            if not bull.empty and not bool(bull.iloc[-1]):
                hrp_w = hrp_w * bear_exposure

        out = pd.Series(0.0, index=pred_pivot.columns)
        out.loc[top_syms] = hrp_w.values
        return out

    return signal_fn


def main():
    print("=" * 75)
    print("PHASE 2 BACKTEST — Cross-sectional top-K + HRP + regime filter")
    print("=" * 75)

    # ---- Load models ----
    print("\n[1/6] Loading Phase 2 year-models...")
    with open('data/models/p2_year_models.pkl', 'rb') as f:
        year_models = pickle.load(f)
    print(f"  Loaded {len(year_models)} year-models")

    # ---- Rebuild panel ----
    print("\n[2/6] Rebuilding feature panel and labels...")
    cs_panel = pd.read_parquet('data/processed/features/CS_PANEL.parquet')
    market_context = pd.read_parquet('data/processed/features/MARKET_CONTEXT.parquet')
    panel = assemble_panel_features(cs_panel, market_context)
    labels = build_label_panel(horizon=5, method='pct_rank')
    X, y = align_panel_features_labels(panel, labels, feature_lag=1)
    print(f"  Aligned panel: {X.shape}")

    # ---- Generate predictions ----
    print("\n[3/6] Generating walk-forward predictions...")
    predictions = predict_panel_with_walk_forward(X, year_models)
    predictions = predictions.dropna()
    print(f"  Total predictions: {len(predictions)}")

    # ---- Load prices and build matrix ----
    print("\n[4/6] Loading prices and regime series...")
    universe = sorted([f.stem for f in Path('data/processed').glob('*.parquet')
                       if f.stem != 'BENCHMARK_INDEX'])
    prices = {s: pd.read_parquet(f'data/processed/{s}.parquet')['Close'] for s in universe}
    price_df = pd.DataFrame(prices).dropna(how='all')
    returns_df = price_df.pct_change()

    nifty = pd.read_parquet('data/processed/BENCHMARK_INDEX.parquet')['Close']
    regime = sma_regime(nifty, window=REGIME_WINDOW)
    print(f"  Universe: {len(universe)} stocks")
    print(f"  Price matrix: {price_df.shape}")
    print(f"  Regime - bullish days: {regime.mean():.1%}")

    backtest_start = '2019-01-01'
    bt_prices = price_df.loc[backtest_start:]
    rebalance_dates = get_rebalance_dates(bt_prices.index, frequency='monthly')
    config = BacktestConfig(initial_capital=1_000_000, slippage_bps=1.0, commission_bps=5.0)
    print(f"  Backtest period: {bt_prices.index[0].date()} to {bt_prices.index[-1].date()}")
    print(f"  Rebalances: {len(rebalance_dates)}")

    # ---- STRATEGY 1: P2 with regime filter ----
    print("\n[5/6] Running Phase 2 strategy WITH regime filter...")
    sig = make_p2_signal_fn(predictions, returns_df, regime,
                            top_k=TOP_K, max_position=MAX_POSITION,
                            use_regime=True)
    p2_results = run_backtest(bt_prices, rebalance_dates, sig, config)
    p2_metrics = compute_metrics(p2_results.equity_curve, label='Phase 2 (with regime filter)')
    print_metrics(p2_metrics)
    print(f"  Avg one-way turnover/rebalance: {p2_results.turnover.mean():.2%}")
    print(f"  Avg fraction invested:          {p2_results.weights.sum(axis=1).mean():.2%}")

    # ---- STRATEGY 2: P2 without regime filter ----
    print("\nRunning Phase 2 strategy WITHOUT regime filter...")
    sig_no_reg = make_p2_signal_fn(predictions, returns_df, regime,
                                    top_k=TOP_K, max_position=MAX_POSITION,
                                    use_regime=False)
    p2_no_reg_results = run_backtest(bt_prices, rebalance_dates, sig_no_reg, config)
    p2_no_reg_metrics = compute_metrics(p2_no_reg_results.equity_curve, label='Phase 2 (no regime filter)')
    print_metrics(p2_no_reg_metrics)

    # ---- STRATEGY 3: Equal-weight 46 baseline ----
    print("\n[6/6] Running equal-weight 46-stock baseline...")
    eq_signal = equal_weight_signal(list(bt_prices.columns))
    eq_results = run_backtest(bt_prices, rebalance_dates, eq_signal, config)
    eq_metrics = compute_metrics(eq_results.equity_curve, label='Equal-Weight 46')
    print_metrics(eq_metrics)

    # ---- COMPARISON ----
    print("\n" + "=" * 75)
    print("COMPARISON")
    print("=" * 75)
    comparison = compare_strategies({
        'Phase 2 (regime)':    p2_results.equity_curve,
        'Phase 2 (no regime)': p2_no_reg_results.equity_curve,
        'Equal-Weight 46':     eq_results.equity_curve,
    })
    print(comparison.to_string())

    # ---- Save ----
    out_dir = Path('data/backtest_results')
    out_dir.mkdir(parents=True, exist_ok=True)
    p2_results.equity_curve.to_csv(out_dir / 'p2_equity.csv')
    p2_no_reg_results.equity_curve.to_csv(out_dir / 'p2_no_reg_equity.csv')
    eq_results.equity_curve.to_csv(out_dir / 'eq46_equity.csv')
    p2_results.weights.to_csv(out_dir / 'p2_weights.csv')
    comparison.to_csv(out_dir / 'p2_comparison.csv')

    print(f"\n[OK] Phase 2 backtest results saved to {out_dir}/")
    print("=" * 75)


if __name__ == '__main__':
    main()
