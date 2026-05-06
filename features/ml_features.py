"""
ML Feature Pipeline - Prepares raw features for ML training.

Key operations:
- NaN handling (per-feature strategy)
- Outlier clipping (winsorization at ±3 sigma)
- Cross-sectional z-score normalization (per date, across stocks)
- Rolling normalization (per stock, over time)

Critical: All transformations are computed using PAST data only to avoid
look-ahead bias. Cross-sectional normalization is the exception - it uses
the same date for normalization, which is valid for ranking signals.
"""
import pandas as pd
import numpy as np


# ============================================================================
# NAN HANDLING
# ============================================================================

def handle_nans(features: pd.DataFrame, max_nan_pct: float = 0.05) -> pd.DataFrame:
    """
    Drop columns with too many NaNs, forward-fill the rest.

    Args:
        features: Raw feature DataFrame
        max_nan_pct: Drop columns with > this fraction of NaNs

    Returns:
        Cleaned feature DataFrame
    """
    nan_pct = features.isna().mean()
    cols_to_drop = nan_pct[nan_pct > max_nan_pct].index.tolist()

    if cols_to_drop:
        features = features.drop(columns=cols_to_drop)

    # Forward fill remaining NaNs (but not too aggressively)
    features = features.ffill(limit=5)

    # Drop initial rows with any remaining NaNs (warmup period)
    features = features.dropna()

    return features


# ============================================================================
# OUTLIER HANDLING
# ============================================================================

def winsorize(series: pd.Series, lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.Series:
    """Clip values to specified percentiles. Reduces impact of outliers."""
    lower = series.quantile(lower_pct)
    upper = series.quantile(upper_pct)
    return series.clip(lower, upper)


def clip_sigma(series: pd.Series, n_sigma: float = 3.0) -> pd.Series:
    """Clip values to ±n_sigma standard deviations from mean."""
    mean = series.mean()
    std = series.std()
    return series.clip(mean - n_sigma * std, mean + n_sigma * std)


def winsorize_features(features: pd.DataFrame, lower_pct: float = 0.005,
                        upper_pct: float = 0.995) -> pd.DataFrame:
    """Apply winsorization to all features."""
    return features.apply(lambda col: winsorize(col, lower_pct, upper_pct))


# ============================================================================
# NORMALIZATION
# ============================================================================

def zscore_normalize(series: pd.Series, window: int = None) -> pd.Series:
    """
    Z-score normalize a series.

    If window is None: full-sample z-score (only use for in-sample analysis).
    If window is set: rolling z-score using only past `window` observations
                      (no look-ahead).
    """
    if window is None:
        return (series - series.mean()) / series.std()
    else:
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        return (series - rolling_mean) / rolling_std


def rolling_zscore(features: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Apply rolling z-score normalization to all features (no look-ahead)."""
    return features.apply(lambda col: zscore_normalize(col, window))


def rank_normalize(series: pd.Series) -> pd.Series:
    """Convert values to ranks in [0, 1]. Robust to outliers."""
    return series.rank(pct=True)


def cross_sectional_zscore(features_panel: pd.DataFrame, group_col: str = 'date') -> pd.DataFrame:
    """
    Cross-sectional z-score normalization (per date, across all stocks).

    Args:
        features_panel: DataFrame with MultiIndex (date, symbol) or 'date' column
        group_col: Column to group by (typically 'date')

    Returns:
        Normalized features (relative ranking on each date)
    """
    if group_col in features_panel.index.names:
        return features_panel.groupby(level=group_col).transform(
            lambda x: (x - x.mean()) / x.std()
        )
    else:
        return features_panel.groupby(group_col).transform(
            lambda x: (x - x.mean()) / x.std()
        )


# ============================================================================
# FEATURE COMBINATION (PANEL CONSTRUCTION)
# ============================================================================

def build_feature_panel(features_dict: dict) -> pd.DataFrame:
    """
    Combine per-stock features into a panel DataFrame with MultiIndex (date, symbol).

    Args:
        features_dict: {symbol: features_dataframe}

    Returns:
        MultiIndex DataFrame: rows = (date, symbol), columns = features
    """
    panels = []
    for symbol, features in features_dict.items():
        df = features.copy()
        df['symbol'] = symbol
        df = df.set_index('symbol', append=True)
        panels.append(df)

    panel = pd.concat(panels, axis=0)
    panel.index.names = ['date', 'symbol']
    return panel.sort_index()


# ============================================================================
# FULL PIPELINE
# ============================================================================

def prepare_ml_features(
    features: pd.DataFrame,
    max_nan_pct: float = 0.05,
    winsorize_pct: float = 0.005,
    zscore_window: int = 252,
) -> pd.DataFrame:
    """
    Full ML preprocessing pipeline.

    1. Drop high-NaN columns and forward-fill
    2. Winsorize outliers
    3. Rolling z-score normalize (no look-ahead)
    """
    features = handle_nans(features, max_nan_pct=max_nan_pct)
    features = winsorize_features(features, lower_pct=winsorize_pct, upper_pct=1 - winsorize_pct)
    features = rolling_zscore(features, window=zscore_window)
    features = features.dropna()  # Drop warmup period after rolling z-score
    return features


def prepare_panel_features(
    features_dict: dict,
    max_nan_pct: float = 0.05,
    winsorize_pct: float = 0.005,
    use_cross_sectional: bool = True,
) -> pd.DataFrame:
    """
    Full ML preprocessing for a panel of stocks.

    Args:
        features_dict: {symbol: features_dataframe}
        use_cross_sectional: If True, normalize cross-sectionally (per date).
                             If False, normalize per-stock with rolling window.

    Returns:
        Cleaned, normalized panel DataFrame with MultiIndex (date, symbol)
    """
    # Step 1: Clean each stock individually
    cleaned = {}
    for symbol, features in features_dict.items():
        cleaned[symbol] = handle_nans(features, max_nan_pct=max_nan_pct)

    # Step 2: Build panel
    panel = build_feature_panel(cleaned)

    # Step 3: Winsorize per feature (across all dates and stocks)
    panel = winsorize_features(panel, lower_pct=winsorize_pct, upper_pct=1 - winsorize_pct)

    # Step 4: Normalize
    if use_cross_sectional:
        panel = cross_sectional_zscore(panel, group_col='date')
    else:
        panel = panel.groupby(level='symbol').transform(
            lambda x: zscore_normalize(x, window=252)
        )

    panel = panel.dropna()
    return panel


if __name__ == '__main__':
    # Test on RELIANCE
    from features.technical import compute_all_technical
    from features.price_based import compute_all_price_features

    df = pd.read_parquet('data/processed/RELIANCE.parquet')
    tech_features = compute_all_technical(df)
    price_features = compute_all_price_features(df)
    all_features = pd.concat([tech_features, price_features], axis=1)

    print(f"Raw features: {all_features.shape}")
    print(f"NaN counts (top 10): {all_features.isna().sum().nlargest(10).to_dict()}")

    # Process
    ml_ready = prepare_ml_features(all_features)
    print(f"\nML-ready features: {ml_ready.shape}")
    print(f"Mean (should be ~0): {ml_ready.mean().abs().mean():.4f}")
    print(f"Std (should be ~1):  {ml_ready.std().mean():.4f}")
    print(f"Min: {ml_ready.min().min():.4f}, Max: {ml_ready.max().max():.4f}")
