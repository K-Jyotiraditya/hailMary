"""
Phase 2.1 Training — LightGBM + Random Forest ensemble, walk-forward.

Trains both models per year on the same panel data, saves both for
ensemble prediction at backtest time.

Run: PYTHONPATH=. python scripts/p21_train.py
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
)
from ml.rf_model import train_rf
from ml.ensemble import ensemble_predict


def main():
    print("=" * 75)
    print("PHASE 2.1 TRAINING — LightGBM + RandomForest ensemble")
    print("=" * 75)

    # Load (already built in Phase 2)
    print("\n[1/4] Loading panel features and labels...")
    cs_panel = pd.read_parquet('data/processed/features/CS_PANEL.parquet')
    market_context = pd.read_parquet('data/processed/features/MARKET_CONTEXT.parquet')
    panel = assemble_panel_features(cs_panel, market_context)
    labels = build_label_panel(horizon=5, method='pct_rank')
    X, y = align_panel_features_labels(panel, labels, feature_lag=1)
    dates = X.index.get_level_values('date')
    print(f"  Panel: {X.shape}")

    # Train walk-forward
    print("\n[2/4] Training LightGBM + RF for each year...")
    backtest_start_year = 2019
    end_year = dates.max().year
    year_models = {}

    for year in range(backtest_start_year, end_year + 1):
        cutoff = pd.Timestamp(f'{year}-01-01')
        train_mask = dates < cutoff
        if train_mask.sum() < 10000:
            print(f"  {year}: skipped ({train_mask.sum()} samples)")
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        print(f"  {year}: training on {train_mask.sum()} samples...", end='', flush=True)

        lgbm = train_cs_model(X_tr, y_tr)
        rf = train_rf(X_tr, y_tr)

        # Evaluate ensemble on the upcoming year
        test_mask = (dates >= cutoff) & (dates < pd.Timestamp(f'{year+1}-01-01'))
        oos_metrics = None
        if test_mask.sum() > 100:
            X_te, y_te = X[test_mask], y[test_mask]
            lgbm_pred = pd.Series(lgbm.predict(X_te), index=X_te.index)
            rf_pred = pd.Series(rf.predict(X_te), index=X_te.index)
            ens = ensemble_predict(lgbm_pred, rf_pred, w_lgbm=0.6, w_rf=0.4, rerank=True)

            metrics_lgbm = evaluate_predictions(y_te, lgbm_pred)
            metrics_rf = evaluate_predictions(y_te, rf_pred)
            metrics_ens = evaluate_predictions(y_te, ens)

            oos_metrics = {
                'lgbm_ic': metrics_lgbm['mean_daily_ic'],
                'rf_ic': metrics_rf['mean_daily_ic'],
                'ensemble_ic': metrics_ens['mean_daily_ic'],
                'ensemble_t': metrics_ens['ic_t_stat'],
                'ensemble_pos_days': metrics_ens['positive_ic_days'],
            }
            print(f" OOS IC: lgbm={metrics_lgbm['mean_daily_ic']:+.4f} "
                  f"rf={metrics_rf['mean_daily_ic']:+.4f} "
                  f"ens={metrics_ens['mean_daily_ic']:+.4f}")
        else:
            print(" (no OOS data)")

        year_models[year] = {
            'lgbm': lgbm,
            'rf': rf,
            'feature_names': X.columns.tolist(),
            'oos_metrics': oos_metrics,
            'train_size': train_mask.sum(),
        }

    # Aggregate
    print("\n[3/4] Aggregate OOS performance:")
    rows = []
    for year, m in year_models.items():
        if m.get('oos_metrics'):
            rows.append({'year': year, **m['oos_metrics']})
    if rows:
        agg = pd.DataFrame(rows)
        print(agg.round(4).to_string(index=False))
        print(f"\n  Mean ensemble IC: {agg['ensemble_ic'].mean():+.4f}")
        print(f"  Years ens > lgbm: {(agg['ensemble_ic'] > agg['lgbm_ic']).sum()}/{len(agg)}")
        print(f"  Years ens > rf:   {(agg['ensemble_ic'] > agg['rf_ic']).sum()}/{len(agg)}")

    # Save
    print("\n[4/4] Saving models...")
    out_dir = Path('data/models')
    out_dir.mkdir(parents=True, exist_ok=True)
    save_payload = {
        year: {
            'lgbm': m['lgbm'],
            'rf': m['rf'],
            'feature_names': m['feature_names'],
            **{k: v for k, v in m.items() if k not in ('lgbm', 'rf', 'feature_names')}
        }
        for year, m in year_models.items()
    }
    with open(out_dir / 'p21_year_models.pkl', 'wb') as f:
        pickle.dump(save_payload, f)
    print(f"  Saved {len(year_models)} year-models to {out_dir}/p21_year_models.pkl")

    print("=" * 75)


if __name__ == '__main__':
    main()
