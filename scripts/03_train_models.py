"""
End-to-end training pipeline: PER-STOCK MODELS WITH CPCV

KEY DESIGN DECISION (validated by diagnostic experiments):
  Pooled (one model for all stocks) gives AUC ≈ 0.50, IC ≈ 0.003 — no signal.
  Per-stock models give mean IC ≈ 0.020, with 8/18 stocks above 0.02 IC.
  Different stocks have different feature->return relationships, and pooling
  averages them out. We train one LightGBM per stock.

Pipeline:
  1. Load ML-ready features for each stock
  2. Generate triple-barrier labels using each stock's Close prices
  3. Align features to labels (lag features by 1 day)
  4. Train LightGBM with CPCV evaluation per stock
  5. Save per-stock models + per-stock metrics + SHAP importance

Run: PYTHONPATH=. python scripts/03_train_models.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from labels.triple_barrier import triple_barrier_labels, get_label_summary
from ml.preprocessor import align_features_labels
from ml.lgbm_model import (
    cpcv_evaluate,
    aggregate_importance,
    make_lgbm_model,
)


# Hyperparameters: less aggressive regularization works better for per-stock training
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


def train_one_stock(symbol: str, features: pd.DataFrame, close: pd.Series) -> dict:
    """Train and evaluate models for a single stock."""
    # Generate triple-barrier labels
    labels = triple_barrier_labels(close, pt_mult=2.0, sl_mult=2.0,
                                   max_holding=5, vol_span=20)
    if labels.empty or len(labels) < 500:
        return None

    # Align features (lagged by 1 day) to labels
    X, y, t1 = align_features_labels(features, labels, feature_lag=1)
    if len(X) < 500:
        return None

    # 5-day forward returns for IC calculation (positionally aligned with X)
    forward_5d = close.pct_change(5).shift(-5).reindex(X.index)

    # CPCV with 6 groups, k=2 = 15 train/test splits
    metrics, models = cpcv_evaluate(
        X, y, t1,
        forward_returns=forward_5d,
        n_groups=6, k_test=2, embargo_pct=0.01,
        params=LGBM_PARAMS,
        n_estimators=N_ESTIMATORS,
    )

    if metrics.empty:
        return None

    # Train final production model on ALL data (for live signal generation)
    final_model = make_lgbm_model(LGBM_PARAMS, N_ESTIMATORS)
    final_model.fit(X, y)

    # Aggregate feature importance across CPCV models
    importance = aggregate_importance(models, X.columns.tolist())

    return {
        'symbol': symbol,
        'metrics': metrics,
        'cpcv_models': models,
        'final_model': final_model,
        'importance': importance,
        'feature_names': X.columns.tolist(),
        'n_train_samples': len(X),
        'label_summary': get_label_summary(labels),
        'mean_auc': metrics['auc'].mean(),
        'std_auc': metrics['auc'].std(),
        'mean_ic': metrics['ic'].mean(),
        'std_ic': metrics['ic'].std(),
    }


def main():
    print("=" * 75)
    print("STEP 3: TRAIN PER-STOCK MODELS WITH CPCV")
    print("=" * 75)

    # Discover all stocks with ML-ready features
    feature_files = sorted(Path('data/processed/features').glob('*_ml.parquet'))
    print(f"\nFound {len(feature_files)} stocks. Training per-stock models...\n")

    all_results = {}
    summary_rows = []

    for fp in feature_files:
        symbol = fp.stem.replace('_ml', '')
        try:
            features = pd.read_parquet(fp)
            close = pd.read_parquet(f'data/processed/{symbol}.parquet')['Close']
            result = train_one_stock(symbol, features, close)

            if result is None:
                print(f"[skip] {symbol}: insufficient data")
                continue

            all_results[symbol] = result
            summary_rows.append({
                'symbol': symbol,
                'n_samples': result['n_train_samples'],
                'mean_auc': result['mean_auc'],
                'std_auc': result['std_auc'],
                'mean_ic': result['mean_ic'],
                'std_ic': result['std_ic'],
                'win_rate': result['label_summary']['win_rate'],
                'class_balance': result['label_summary']['class_balance'].get(1, 0),
            })

            tradeable = "TRADEABLE" if result['mean_ic'] > 0.02 and result['mean_auc'] > 0.51 else "         "
            print(f"  [{tradeable}] {symbol:12} | n={result['n_train_samples']:5d} | "
                  f"AUC={result['mean_auc']:.4f} (+-{result['std_auc']:.3f}) | "
                  f"IC={result['mean_ic']:+.4f}")

        except Exception as e:
            print(f"[err] {symbol}: {type(e).__name__}: {str(e)[:60]}")

    # Aggregate summary
    summary_df = pd.DataFrame(summary_rows).set_index('symbol')

    print("\n" + "=" * 75)
    print("CROSS-STOCK PERFORMANCE SUMMARY")
    print("=" * 75)
    print(summary_df.round(4).to_string())

    print(f"\n  Mean AUC across stocks:      {summary_df['mean_auc'].mean():.4f}")
    print(f"  Mean IC across stocks:       {summary_df['mean_ic'].mean():.4f}")
    print(f"  Stocks with IC > 0.02:       {(summary_df['mean_ic'] > 0.02).sum()}/{len(summary_df)}")
    print(f"  Stocks with AUC > 0.52:      {(summary_df['mean_auc'] > 0.52).sum()}/{len(summary_df)}")
    print(f"  TRADEABLE (AUC>0.51 & IC>0.02): "
          f"{((summary_df['mean_auc'] > 0.51) & (summary_df['mean_ic'] > 0.02)).sum()}/{len(summary_df)}")

    # Identify tradeable universe
    tradeable_mask = (summary_df['mean_auc'] > 0.51) & (summary_df['mean_ic'] > 0.02)
    tradeable_stocks = summary_df[tradeable_mask].index.tolist()
    print(f"\nTRADEABLE UNIVERSE ({len(tradeable_stocks)} stocks):")
    print(f"  {tradeable_stocks}")

    # Save artifacts
    out_dir = Path('data/models')
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(out_dir / 'per_stock_metrics.csv')
    pd.Series(tradeable_stocks).to_csv(out_dir / 'tradeable_universe.csv', index=False, header=['symbol'])

    # Save all per-stock results in one pickle (final models + cpcv models)
    save_payload = {
        symbol: {
            'final_model': r['final_model'],
            'cpcv_models': r['cpcv_models'],
            'feature_names': r['feature_names'],
            'metrics': r['metrics'],
            'importance': r['importance'],
        }
        for symbol, r in all_results.items()
    }
    with open(out_dir / 'per_stock_models.pkl', 'wb') as f:
        pickle.dump(save_payload, f)

    # Aggregate top features across tradeable stocks
    if tradeable_stocks:
        all_imp = []
        for s in tradeable_stocks:
            imp = all_results[s]['importance']
            all_imp.append(imp['mean_importance'].rename(s))
        cross_imp = pd.concat(all_imp, axis=1)
        avg_imp = cross_imp.mean(axis=1).sort_values(ascending=False)
        print(f"\nTop 15 features across tradeable stocks (avg gain importance):")
        print(avg_imp.head(15).round(2).to_string())
        avg_imp.to_csv(out_dir / 'tradeable_feature_importance.csv', header=['mean_importance'])

    print(f"\n[OK] Artifacts saved to {out_dir}/")
    print("=" * 75)


if __name__ == '__main__':
    main()
