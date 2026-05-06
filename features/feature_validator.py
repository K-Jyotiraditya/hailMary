"""
Feature Validator - Quality checks for engineered features.

Validates:
- Stationarity (ADF test) - features should be stationary for ML
- Correlation - identify and flag multicollinear feature pairs
- Information Coefficient (IC) - signal-vs-future-return correlation
- Look-ahead bias detection - regression test for accidental future leakage
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from typing import List, Tuple


# ============================================================================
# STATIONARITY
# ============================================================================

def adf_test(series: pd.Series) -> dict:
    """
    Augmented Dickey-Fuller test. Returns p-value and stationarity flag.
    Null hypothesis: series has a unit root (non-stationary).
    Reject if p < 0.05 → series is stationary.
    """
    s = series.dropna()
    if len(s) < 50 or s.std() == 0:
        return {'p_value': np.nan, 'is_stationary': False}

    try:
        result = adfuller(s, autolag='AIC')
        return {
            'p_value': result[1],
            'is_stationary': result[1] < 0.05,
        }
    except Exception:
        return {'p_value': np.nan, 'is_stationary': False}


def check_stationarity(features: pd.DataFrame) -> pd.DataFrame:
    """Check stationarity of all features. Returns summary DataFrame."""
    results = []
    for col in features.columns:
        test = adf_test(features[col])
        results.append({
            'feature': col,
            'adf_p': test['p_value'],
            'is_stationary': test['is_stationary'],
        })
    return pd.DataFrame(results).set_index('feature')


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def find_correlated_pairs(features: pd.DataFrame, threshold: float = 0.95) -> List[Tuple[str, str, float]]:
    """
    Find feature pairs with correlation above threshold.
    These are candidates for removal (multicollinearity reduces model stability).
    """
    corr = features.corr().abs()
    # Upper triangle only (to avoid duplicates)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    pairs = []
    for col in upper.columns:
        high_corr = upper[col][upper[col] > threshold]
        for other_col, value in high_corr.items():
            pairs.append((col, other_col, value))

    return sorted(pairs, key=lambda x: -x[2])


def remove_correlated_features(features: pd.DataFrame, threshold: float = 0.95,
                               importance: pd.Series = None) -> pd.DataFrame:
    """
    Remove highly correlated features. If `importance` is provided, keep
    the more important one in each correlated pair.
    """
    corr = features.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        if col in to_drop:
            continue
        for other_col in upper.index:
            if other_col in to_drop or other_col == col:
                continue
            if upper.loc[other_col, col] > threshold:
                # Drop the less important one
                if importance is not None:
                    if importance.get(col, 0) >= importance.get(other_col, 0):
                        to_drop.add(other_col)
                    else:
                        to_drop.add(col)
                        break
                else:
                    to_drop.add(other_col)

    return features.drop(columns=list(to_drop))


# ============================================================================
# INFORMATION COEFFICIENT (IC)
# ============================================================================

def information_coefficient(feature: pd.Series, forward_return: pd.Series,
                           method: str = 'spearman') -> float:
    """
    Information Coefficient: correlation between feature and forward return.

    Spearman (rank correlation) is robust to outliers and preferred in finance.
    Pearson is appropriate when both are normally distributed.

    Interpretation:
    - |IC| > 0.1: Very strong signal (rare in equity markets)
    - |IC| > 0.05: Strong signal
    - |IC| > 0.02: Useful signal
    - |IC| < 0.02: Weak/noise
    """
    aligned = pd.concat([feature, forward_return], axis=1).dropna()
    if len(aligned) < 30:
        return np.nan
    return aligned.corr(method=method).iloc[0, 1]


def compute_ic_summary(features: pd.DataFrame, close: pd.Series,
                       horizon: int = 5) -> pd.DataFrame:
    """
    Compute IC for every feature against forward returns at given horizon.
    """
    forward_return = close.pct_change(horizon).shift(-horizon)

    ic_results = []
    for col in features.columns:
        ic = information_coefficient(features[col], forward_return, method='spearman')
        ic_results.append({
            'feature': col,
            f'ic_{horizon}d': ic,
            f'abs_ic_{horizon}d': abs(ic) if not np.isnan(ic) else np.nan,
        })

    return pd.DataFrame(ic_results).set_index('feature').sort_values(
        f'abs_ic_{horizon}d', ascending=False
    )


# ============================================================================
# LOOK-AHEAD BIAS DETECTION
# ============================================================================

def detect_lookahead_bias(features: pd.DataFrame, close: pd.Series,
                          threshold: float = 0.5) -> List[str]:
    """
    Detect features that may have look-ahead bias.

    A feature with very high correlation (|r| > threshold) to SAME-DAY return
    is suspicious. Future-return leakage typically shows |r| > 0.5 with
    contemporaneous returns, which is rare for legitimate features.
    """
    same_day_return = close.pct_change()
    suspicious = []

    for col in features.columns:
        try:
            corr = features[col].corr(same_day_return)
            if abs(corr) > threshold:
                suspicious.append((col, corr))
        except Exception:
            pass

    return suspicious


# ============================================================================
# COMPREHENSIVE VALIDATION
# ============================================================================

def validate_features(features: pd.DataFrame, close: pd.Series,
                     verbose: bool = True) -> dict:
    """
    Run full validation suite on features.

    Returns dictionary with all validation results.
    """
    results = {}

    # 1. Basic stats
    results['n_features'] = features.shape[1]
    results['n_observations'] = features.shape[0]
    results['nan_pct'] = features.isna().mean().to_dict()

    # 2. Stationarity
    stationarity = check_stationarity(features)
    results['stationary_count'] = stationarity['is_stationary'].sum()
    results['non_stationary_features'] = stationarity[~stationarity['is_stationary']].index.tolist()

    # 3. Correlation pairs
    correlated_pairs = find_correlated_pairs(features, threshold=0.95)
    results['correlated_pairs'] = correlated_pairs[:20]  # Top 20

    # 4. Information Coefficient (5-day horizon)
    ic_summary = compute_ic_summary(features, close, horizon=5)
    results['top_ic_features'] = ic_summary.head(10).to_dict()
    results['mean_abs_ic'] = ic_summary['abs_ic_5d'].mean()

    # 5. Look-ahead detection
    suspicious = detect_lookahead_bias(features, close, threshold=0.5)
    results['suspicious_features'] = suspicious

    if verbose:
        print(f"\n{'=' * 70}")
        print("FEATURE VALIDATION REPORT")
        print(f"{'=' * 70}")
        print(f"Features: {results['n_features']}, Observations: {results['n_observations']}")
        print(f"\nStationarity: {results['stationary_count']}/{results['n_features']} stationary")
        if results['non_stationary_features']:
            print(f"Non-stationary: {', '.join(results['non_stationary_features'][:10])}")

        print(f"\nHighly correlated pairs (|r| > 0.95): {len(correlated_pairs)}")
        for pair in correlated_pairs[:5]:
            print(f"  {pair[0]} <-> {pair[1]}: {pair[2]:.3f}")

        print(f"\nMean |IC| (5-day forward returns): {results['mean_abs_ic']:.4f}")
        print(f"\nTop 10 features by |IC|:")
        print(ic_summary.head(10).to_string())

        if suspicious:
            print(f"\n[!] Suspicious (possible look-ahead): {len(suspicious)}")
            for feat, corr in suspicious[:5]:
                print(f"  {feat}: corr_with_same_day_return = {corr:.3f}")
        else:
            print(f"\n[OK] No look-ahead bias detected")

    return results


if __name__ == '__main__':
    from features.technical import compute_all_technical
    from features.price_based import compute_all_price_features
    from features.ml_features import handle_nans

    df = pd.read_parquet('data/processed/RELIANCE.parquet')
    tech = compute_all_technical(df)
    price = compute_all_price_features(df)
    features = pd.concat([tech, price], axis=1)
    features = handle_nans(features, max_nan_pct=0.10)

    validate_features(features, df['Close'], verbose=True)
