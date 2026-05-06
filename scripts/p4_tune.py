"""
Phase 4 Tuning — find optimal LightGBM hyperparameters with Optuna,
then combine with feature selection and multi-horizon ensembling.

Run: PYTHONPATH=. python scripts/p4_tune.py
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
    assemble_panel_features, align_panel_features_labels, evaluate_predictions,
)
from ml.feature_selection import (
    feature_yearly_ic, score_features, select_features,
)
from ml.optuna_tuner import tune_lgbm


def main():
    print("=" * 75)
    print("PHASE 4 — Hyperparameter tuning + feature selection")
    print("=" * 75)

    # Load panel
    print("\n[1/3] Loading panel...")
    cs_panel = pd.read_parquet('data/processed/features/CS_PANEL.parquet')
    market_context = pd.read_parquet('data/processed/features/MARKET_CONTEXT.parquet')
    panel = assemble_panel_features(cs_panel, market_context)
    labels = build_label_panel(horizon=5, method='pct_rank')
    X, y = align_panel_features_labels(panel, labels, feature_lag=1)
    print(f"  Panel: {X.shape}")

    # Feature selection on training period only (avoid leakage)
    # Use 2019-2024 for IC scoring, leaving 2025-2026 untouched for final test
    print("\n[2/3] Feature selection (IC stability across years)...")
    dates = X.index.get_level_values('date')
    train_mask = dates < pd.Timestamp('2025-01-01')
    yearly = feature_yearly_ic(X[train_mask], y[train_mask])
    scores = score_features(yearly)
    selected = select_features(scores, min_mean_abs_ic=0.005,
                                min_hit_rate=0.55, top_n=80)
    print(f"  Selected {len(selected)}/{X.shape[1]} features")
    print(f"  Top 10: {selected[:10]}")

    X_sel = X[selected]

    # Optuna tuning on a SUBSET of training data (for speed) using WF folds
    print(f"\n[3/3] Tuning LightGBM on {X_sel.shape} (selected features)...")
    print(f"  Running Optuna with 30 trials, 3-fold WF validation...\n")

    # Limit data range so tuning runs in reasonable time
    train_for_tuning = X_sel[dates < pd.Timestamp('2025-01-01')]
    y_for_tuning = y[dates < pd.Timestamp('2025-01-01')]

    tune_result = tune_lgbm(train_for_tuning, y_for_tuning,
                            n_trials=30, n_folds=3, seed=42)

    print(f"\n{'=' * 75}")
    print(f"BEST PARAMETERS FOUND:")
    print(f"{'=' * 75}")
    for k, v in tune_result['best_params'].items():
        if isinstance(v, float):
            print(f"  {k:25} = {v:.5f}")
        else:
            print(f"  {k:25} = {v}")
    print(f"\n  Best CV mean OOS IC: {tune_result['best_ic']:.4f}")

    # Save best params + selected features
    out_dir = Path('data/models')
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'best_params': tune_result['best_params'],
        'best_ic': tune_result['best_ic'],
        'selected_features': selected,
        'feature_scores': scores,
    }
    with open(out_dir / 'p4_tuning.pkl', 'wb') as f:
        pickle.dump(payload, f)

    print(f"\n[OK] Saved tuning results to {out_dir}/p4_tuning.pkl")
    print("=" * 75)


if __name__ == '__main__':
    main()
