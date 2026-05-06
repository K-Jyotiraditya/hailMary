"""
Feature Engineering Pipeline - Compute features for all NIFTY 20 stocks.

Pipeline:
1. Load OHLCV from data/processed/*.parquet
2. Compute technical indicators (35 features)
3. Compute price-based features (27 features)
4. Drop high-NaN columns, forward-fill
5. Save raw features to data/processed/features/{symbol}_raw.parquet
6. Apply ML preprocessing (winsorize + rolling z-score)
7. Save ML-ready features to data/processed/features/{symbol}_ml.parquet

Run from project root: python -m scripts.02_engineer_features
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from features.technical import compute_all_technical
from features.price_based import compute_all_price_features
from features.ml_features import handle_nans, winsorize_features, rolling_zscore
from features.feature_validator import (
    check_stationarity,
    find_correlated_pairs,
    compute_ic_summary,
    detect_lookahead_bias,
)


def engineer_features_one_stock(df: pd.DataFrame) -> tuple:
    """
    Compute all features for one stock.

    Returns (raw_features, ml_features) tuple.
    """
    # Compute technical and price-based features
    tech = compute_all_technical(df)
    price = compute_all_price_features(df)
    raw = pd.concat([tech, price], axis=1)

    # Clean NaNs
    raw = handle_nans(raw, max_nan_pct=0.10)

    # ML preprocessing: winsorize + rolling z-score (no look-ahead)
    ml = winsorize_features(raw, lower_pct=0.005, upper_pct=0.995)
    ml = rolling_zscore(ml, window=252)
    ml = ml.dropna()

    return raw, ml


def main():
    input_dir = Path('data/processed')
    output_dir = Path('data/processed/features')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find OHLCV parquet files (exclude features/)
    parquet_files = [f for f in input_dir.glob('*.parquet') if 'feature' not in f.parent.name]

    print("=" * 70)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 70)
    print(f"\nProcessing {len(parquet_files)} stocks...\n")

    all_ic_summaries = []
    all_validation = []

    for parquet_file in sorted(parquet_files):
        symbol = parquet_file.stem

        try:
            df = pd.read_parquet(parquet_file)

            # Engineer features
            raw, ml = engineer_features_one_stock(df)

            # Save
            raw.to_parquet(output_dir / f"{symbol}_raw.parquet")
            ml.to_parquet(output_dir / f"{symbol}_ml.parquet")

            # Quick IC summary
            ic = compute_ic_summary(raw, df['Close'], horizon=5)
            top_ic = ic.head(3).index.tolist()
            mean_ic = ic['abs_ic_5d'].mean()

            # Look-ahead check
            suspicious = detect_lookahead_bias(raw, df['Close'], threshold=0.5)

            print(f"[OK] {symbol:15} | raw={raw.shape} | ml={ml.shape} | "
                  f"|IC|={mean_ic:.3f} | top: {top_ic[0][:18]}")

            if suspicious:
                print(f"     [!] Suspicious features: {[s[0] for s in suspicious]}")

            all_ic_summaries.append(ic.assign(symbol=symbol))
            all_validation.append({
                'symbol': symbol,
                'n_features': raw.shape[1],
                'n_obs_raw': raw.shape[0],
                'n_obs_ml': ml.shape[0],
                'mean_abs_ic': mean_ic,
                'top_feature': top_ic[0] if top_ic else None,
                'suspicious_count': len(suspicious),
            })

        except Exception as e:
            print(f"[ERR] {symbol:15} | {type(e).__name__}: {str(e)[:60]}")

    # Aggregate cross-stock IC
    print("\n" + "=" * 70)
    print("CROSS-STOCK FEATURE QUALITY")
    print("=" * 70)

    summary_df = pd.DataFrame(all_validation)
    print(f"\nProcessed: {len(summary_df)} stocks")
    print(f"Mean |IC| across stocks: {summary_df['mean_abs_ic'].mean():.4f}")
    print(f"Mean |IC| std-dev:       {summary_df['mean_abs_ic'].std():.4f}")

    # Top features by average |IC| across all stocks
    if all_ic_summaries:
        combined_ic = pd.concat(all_ic_summaries, axis=0)
        avg_ic_by_feature = combined_ic.groupby('feature')['abs_ic_5d'].agg(['mean', 'std', 'count'])
        avg_ic_by_feature = avg_ic_by_feature.sort_values('mean', ascending=False)

        print(f"\nTop 15 features by average |IC| across all stocks:")
        print(avg_ic_by_feature.head(15).round(4).to_string())

        # Save aggregate
        avg_ic_by_feature.to_csv(output_dir / 'cross_stock_ic_summary.csv')
        summary_df.to_csv(output_dir / 'feature_engineering_summary.csv', index=False)
        print(f"\n[OK] Summaries saved to {output_dir}/")

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
