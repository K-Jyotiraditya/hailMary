"""
End-to-end backtest with WALK-FORWARD RETRAINING (no look-ahead).

CRITICAL DESIGN: The first version of this script used `final_model` which
was trained on ALL data, including the backtest period. That gave Sharpe 1.70
— too good to be true. The hidden problem: the model had already seen the
test data during training.

THIS VERSION uses honest walk-forward:
  - At the start of each year, retrain a fresh model per stock using ONLY
    data available before that year.
  - Use that model to generate signals for the entire upcoming year.
  - Repeat.

This is computationally heavier (~60 retrainings: 10 years × 6 stocks at 250
trees) but it's the only honest way to backtest a strategy that uses ML.

Pipeline:
  1. Load tradeable universe (6 stocks from Week 3)
  2. For each year of the backtest, retrain models on data through year-end - 1.
  3. Generate signals month-by-month using the year-appropriate models.
  4. Run backtest engine with realistic costs.
  5. Compare against equal-weight benchmarks.

Run: PYTHONPATH=. python scripts/04_backtest.py
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from labels.triple_barrier import triple_barrier_labels
from ml.preprocessor import align_features_labels
from ml.lgbm_model import make_lgbm_model
from portfolio.optimizer import hrp_weights_robust
from portfolio.position_sizer import compute_target_weights
from portfolio.rebalancer import get_rebalance_dates
from backtest.engine import run_backtest, BacktestConfig, equal_weight_signal
from backtest.results import compute_metrics, print_metrics, compare_strategies


LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.03,
    'num_leaves': 15,
    'max_depth': 4,
    'min_child_samples': 30,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.05,
    'reg_lambda': 0.05,
    'verbosity': -1,
    'random_state': 42,
}
N_ESTIMATORS = 250


def load_universe_data():
    """Load tradeable universe + features + close prices."""
    universe_df = pd.read_csv('data/models/tradeable_universe.csv')
    tradeable = universe_df['symbol'].tolist()

    features = {}
    prices = {}
    for symbol in tradeable:
        features[symbol] = pd.read_parquet(f'data/processed/features/{symbol}_ml.parquet')
        prices[symbol] = pd.read_parquet(f'data/processed/{symbol}.parquet')['Close']

    return tradeable, features, prices


def train_walk_forward_models(tradeable, features, prices, train_through: pd.Timestamp,
                              min_samples: int = 500) -> dict:
    """
    Train one LightGBM per stock using ONLY data with entry_ts <= train_through.

    Args:
        train_through: Cutoff timestamp; only labels with t1 <= train_through used.
                       This ensures no future leakage.

    Returns:
        {symbol: (model, feature_names)}
    """
    models = {}
    for symbol in tradeable:
        labels = triple_barrier_labels(prices[symbol], pt_mult=2.0, sl_mult=2.0,
                                       max_holding=5, vol_span=20)
        if labels.empty:
            continue

        # Honest cutoff: drop any sample whose label horizon extends past train_through
        labels = labels[labels['t1'] <= train_through]

        X, y, _ = align_features_labels(features[symbol], labels, feature_lag=1)
        if len(X) < min_samples:
            continue

        model = make_lgbm_model(LGBM_PARAMS, N_ESTIMATORS)
        model.fit(X, y)
        models[symbol] = {'model': model, 'feature_names': X.columns.tolist()}

    return models


def make_signal_function_wf(tradeable, features, prices,
                            year_models: dict,
                            threshold=0.50, cov_window=60, sizing_mode='binary'):
    """
    Build a signal function using year-specific walk-forward models.

    Args:
        year_models: {year: {symbol: {model, feature_names}}}
                     For each rebalance date, we use models[date.year].
    """
    returns = pd.DataFrame({s: prices[s].pct_change() for s in tradeable})

    def signal_fn(date):
        year = date.year
        if year not in year_models:
            # Use the most recent available year if data exists
            available = [y for y in year_models if y <= year]
            if not available:
                return pd.Series(0.0, index=tradeable)
            year = max(available)

        active_models = year_models[year]
        if not active_models:
            return pd.Series(0.0, index=tradeable)

        # Compute current probabilities (features at end of `date`, lagged 1)
        probs_today = pd.Series(dtype=float)
        for s in tradeable:
            if s not in active_models:
                continue
            f_lagged = features[s].shift(1)
            if date not in f_lagged.index:
                # Use latest feature row up to date
                f_lagged = f_lagged.loc[:date]
                if f_lagged.empty:
                    continue
                latest = f_lagged.iloc[[-1]]
            else:
                latest = f_lagged.loc[[date]]
            latest = latest.dropna()
            if latest.empty:
                continue
            feat_names = active_models[s]['feature_names']
            prob = active_models[s]['model'].predict_proba(latest[feat_names])[0, 1]
            probs_today[s] = prob

        if probs_today.empty:
            return pd.Series(0.0, index=tradeable)

        # HRP weights from past returns
        ret_window = returns.loc[:date].tail(cov_window).dropna(how='any')
        if len(ret_window) < cov_window // 2:
            hrp_w = pd.Series(1.0 / len(tradeable), index=tradeable)
        else:
            hrp_w = hrp_weights_robust(ret_window[tradeable])

        return compute_target_weights(
            probs_today, hrp_w,
            threshold=threshold,
            mode=sizing_mode,
            max_position=0.40,
        )

    return signal_fn


def main():
    print("=" * 75)
    print("STEP 4: WALK-FORWARD BACKTEST OF THE ML STRATEGY")
    print("=" * 75)

    # Load data
    print("\n[1/5] Loading universe and data...")
    tradeable, features, prices = load_universe_data()
    print(f"  Tradeable universe: {tradeable}")

    # Build the price matrix
    price_df = pd.DataFrame({s: prices[s] for s in tradeable}).dropna()
    print(f"  Price matrix: {price_df.shape}")
    print(f"  Date range:   {price_df.index.min().date()} to {price_df.index.max().date()}")

    # SBILIFE listed late (2017), so backtest starts 2019 to give 2 years of training
    backtest_start = '2019-01-01'
    bt_prices = price_df.loc[backtest_start:]
    rebalance_dates = get_rebalance_dates(bt_prices.index, frequency='monthly')
    print(f"  Backtest period: {backtest_start} -> {bt_prices.index[-1].date()}")
    print(f"  Rebalance count: {len(rebalance_dates)}")

    # Train walk-forward models: one per year, using data BEFORE that year
    print("\n[2/5] Training walk-forward models (one per year, NO look-ahead)...")
    backtest_years = sorted(set(d.year for d in bt_prices.index))
    year_models = {}
    for year in backtest_years:
        cutoff = pd.Timestamp(f'{year}-01-01')
        models = train_walk_forward_models(tradeable, features, prices, train_through=cutoff)
        year_models[year] = models
        print(f"  {year}: trained {len(models)}/{len(tradeable)} stocks (cutoff: {cutoff.date()})")

    # ===== STRATEGY 1: ML walk-forward =====
    print("\n[3/5] Running ML walk-forward strategy...")
    signal_fn = make_signal_function_wf(
        tradeable, features, prices, year_models,
        threshold=0.50, cov_window=60, sizing_mode='binary',
    )
    config = BacktestConfig(initial_capital=1_000_000, slippage_bps=1.0, commission_bps=5.0)
    ml_results = run_backtest(bt_prices, rebalance_dates, signal_fn, config)
    ml_metrics = compute_metrics(ml_results.equity_curve, label='ML Strategy (walk-forward)')
    print_metrics(ml_metrics)
    print(f"\n  Avg one-way turnover/rebalance: {ml_results.turnover.mean():.2%}")
    print(f"  Total cost paid: ${ml_results.trades['cost_dollars'].sum():,.2f}")
    print(f"  Avg fraction invested: {(ml_results.weights.sum(axis=1)).mean():.2%}")

    # ===== STRATEGY 2: Equal-weight 6 =====
    print("\n[4/5] Running equal-weight benchmark (6 stocks)...")
    eq_signal = equal_weight_signal(tradeable)
    eq_results = run_backtest(bt_prices, rebalance_dates, eq_signal, config)
    eq_metrics = compute_metrics(eq_results.equity_curve, label='Equal-Weight 6')
    print_metrics(eq_metrics)

    # ===== STRATEGY 3: All-18 baseline =====
    print("\n[5/5] Running all-18 equal-weight benchmark...")
    all_18 = sorted([f.stem.replace('_ml', '')
                     for f in Path('data/processed/features').glob('*_ml.parquet')])
    full_prices = {}
    for s in all_18:
        try:
            full_prices[s] = pd.read_parquet(f'data/processed/{s}.parquet')['Close']
        except Exception:
            pass
    full_price_df = pd.DataFrame(full_prices).loc[backtest_start:].dropna()
    full_rebalance = get_rebalance_dates(full_price_df.index, frequency='monthly')
    full_signal = equal_weight_signal(list(full_price_df.columns))
    full_results = run_backtest(full_price_df, full_rebalance, full_signal, config)
    full_metrics = compute_metrics(full_results.equity_curve, label='Equal-Weight 18 (NIFTY-ish)')
    print_metrics(full_metrics)

    # ===== COMPARISON =====
    print("\n" + "=" * 75)
    print("COMPARISON")
    print("=" * 75)
    comparison = compare_strategies({
        'ML Strategy (WF)': ml_results.equity_curve,
        'Equal-Weight 6':  eq_results.equity_curve,
        'Equal-Weight 18': full_results.equity_curve,
    })
    print(comparison.to_string())

    # Save artifacts
    out_dir = Path('data/backtest_results')
    out_dir.mkdir(parents=True, exist_ok=True)
    ml_results.equity_curve.to_csv(out_dir / 'ml_wf_equity.csv')
    eq_results.equity_curve.to_csv(out_dir / 'eq6_equity.csv')
    full_results.equity_curve.to_csv(out_dir / 'eq18_equity.csv')
    ml_results.weights.to_csv(out_dir / 'ml_wf_weights.csv')
    ml_results.trades.to_csv(out_dir / 'ml_wf_trades.csv')
    comparison.to_csv(out_dir / 'comparison_walk_forward.csv')

    # Persist year_models for later inspection
    save_payload = {
        year: {
            s: {'model': m['model'], 'feature_names': m['feature_names']}
            for s, m in ms.items()
        }
        for year, ms in year_models.items()
    }
    with open(out_dir / 'year_models.pkl', 'wb') as f:
        pickle.dump(save_payload, f)

    print(f"\n[OK] Walk-forward backtest results saved to {out_dir}/")
    print("=" * 75)


if __name__ == '__main__':
    main()
