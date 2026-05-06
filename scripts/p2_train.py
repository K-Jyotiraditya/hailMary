"""
Phase 2 Training: cross-sectional rank model with walk-forward retraining.

Pipeline:
  1. Load per-stock features (already computed by 02_engineer_features.py).
  2. Stack into panel + add cross-sectional ranks.
  3. Join market-context features (broadcast to each stock).
  4. Build cross-sectional rank labels (forward 5d return, percentile).
  5. Walk-forward retrain (one LightGBM regressor per year, using only past data).
  6. Save year-models for the backtest.

Run: PYTHONPATH=. python scripts/p2_train.py
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
    train_cs_model,
    evaluate_predictions,
    feature_importance,
)


def main():
    print("=" * 75)
    print("PHASE 2 TRAINING — cross-sectional rank model with walk-forward")
    print("=" * 75)

    # ---- Build feature panel ----
    print("\n[1/5] Building cross-sectional feature panel...")
    cs_panel_path = Path('data/processed/features/CS_PANEL.parquet')
    cs_panel = build_cross_sectional_features(method='rank', save_path=str(cs_panel_path))

    print("\n[2/5] Building market-context features...")
    mc_path = Path('data/processed/features/MARKET_CONTEXT.parquet')
    if mc_path.exists():
        market_context = pd.read_parquet(mc_path)
        print(f"  Loaded existing: {market_context.shape}")
    else:
        market_context = build_market_context()

    print("\n[3/5] Assembling full panel and labels...")
    panel = assemble_panel_features(cs_panel, market_context)
    print(f"  Combined panel: {panel.shape} (features include cs_* and mkt_*)")

    labels = build_label_panel(horizon=5, method='pct_rank')
    print(f"  Labels: {labels.shape}")

    X, y = align_panel_features_labels(panel, labels, feature_lag=1)
    print(f"  Aligned (X, y): {X.shape}, {y.shape}")
    print(f"  Date range: {X.index.get_level_values('date').min().date()} to "
          f"{X.index.get_level_values('date').max().date()}")

    # ---- Walk-forward training ----
    print("\n[4/5] Walk-forward training: one model per year...")
    dates = X.index.get_level_values('date')
    backtest_start_year = 2019
    end_year = dates.max().year

    year_models = {}
    for year in range(backtest_start_year, end_year + 1):
        cutoff = pd.Timestamp(f'{year}-01-01')
        train_mask = dates < cutoff

        if train_mask.sum() < 10000:
            print(f"  {year}: skipped ({train_mask.sum()} samples — too few)")
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        model = train_cs_model(X_tr, y_tr)

        # Evaluate on the year that's now coming up (out-of-sample)
        test_mask = (dates >= cutoff) & (dates < pd.Timestamp(f'{year+1}-01-01'))
        if test_mask.sum() > 100:
            X_te, y_te = X[test_mask], y[test_mask]
            preds = model.predict(X_te)
            metrics = evaluate_predictions(y_te, preds)
            year_models[year] = {
                'model': model,
                'feature_names': X.columns.tolist(),
                'oos_metrics': metrics,
                'train_size': train_mask.sum(),
                'test_size': test_mask.sum(),
            }
            print(f"  {year}: trained on {train_mask.sum():>6} | "
                  f"OOS daily IC = {metrics['mean_daily_ic']:+.4f} "
                  f"(t={metrics['ic_t_stat']:+.2f}, "
                  f"{metrics['positive_ic_days']:.1%} positive days)")
        else:
            year_models[year] = {
                'model': model,
                'feature_names': X.columns.tolist(),
                'train_size': train_mask.sum(),
            }
            print(f"  {year}: trained on {train_mask.sum()} (no OOS year to evaluate yet)")

    # ---- Aggregate IC summary ----
    print("\n[5/5] Aggregate OOS performance:")
    ics = [m['oos_metrics']['mean_daily_ic'] for m in year_models.values() if 'oos_metrics' in m]
    pos_days = [m['oos_metrics']['positive_ic_days'] for m in year_models.values() if 'oos_metrics' in m]
    if ics:
        print(f"  Mean daily IC across years: {np.mean(ics):+.4f} +- {np.std(ics):.4f}")
        print(f"  Mean positive-IC days:      {np.mean(pos_days):.1%}")
        print(f"  Years with positive IC:     {sum(1 for ic in ics if ic > 0)}/{len(ics)}")

    # ---- Top features (last year's model) ----
    last_year = max(year_models.keys())
    if 'feature_names' in year_models[last_year]:
        imp = feature_importance(year_models[last_year]['model'], year_models[last_year]['feature_names'])
        print(f"\nTop 20 features ({last_year} model gain importance):")
        print(imp.head(20).round(2).to_string())

    # ---- Save artifacts ----
    out_dir = Path('data/models')
    out_dir.mkdir(parents=True, exist_ok=True)

    save_payload = {
        year: {
            'model': m['model'],
            'feature_names': m['feature_names'],
            **{k: v for k, v in m.items() if k not in ('model', 'feature_names')}
        }
        for year, m in year_models.items()
    }
    with open(out_dir / 'p2_year_models.pkl', 'wb') as f:
        pickle.dump(save_payload, f)

    summary = pd.DataFrame({
        year: {
            'train_size': m.get('train_size'),
            'test_size': m.get('test_size'),
            'oos_ic': m.get('oos_metrics', {}).get('mean_daily_ic'),
            'oos_t_stat': m.get('oos_metrics', {}).get('ic_t_stat'),
            'positive_ic_days': m.get('oos_metrics', {}).get('positive_ic_days'),
        }
        for year, m in year_models.items()
    }).T
    summary.to_csv(out_dir / 'p2_year_metrics.csv')

    print(f"\n[OK] Saved {len(year_models)} year-models to {out_dir}/p2_year_models.pkl")
    print("=" * 75)


if __name__ == '__main__':
    main()
