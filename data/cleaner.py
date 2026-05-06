"""
Data Cleaner - Handle missing data, corporate actions, and data validation.
"""
import pandas as pd
import numpy as np
from typing import Tuple


def forward_fill_prices(df: pd.DataFrame, method: str = 'ffill', limit: int = 5) -> pd.DataFrame:
    """
    Forward fill missing prices with limit.

    Args:
        df: OHLCV DataFrame
        method: Fill method ('ffill' or 'bfill')
        limit: Max consecutive fill days

    Returns:
        Cleaned DataFrame
    """
    price_cols = ['Open', 'High', 'Low', 'Close']
    volume_cols = ['Volume']

    for col in price_cols:
        if col in df.columns:
            if method == 'ffill':
                df[col] = df[col].ffill(limit=limit)
            elif method == 'bfill':
                df[col] = df[col].bfill(limit=limit)

    for col in volume_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def handle_outliers(df: pd.DataFrame, column: str = 'Close', threshold: float = 5.0) -> Tuple[pd.DataFrame, list]:
    """
    Detect and flag outliers in price data.

    Args:
        df: OHLCV DataFrame
        column: Column to check (typically 'Close')
        threshold: Z-score threshold (5 sigma)

    Returns:
        (Cleaned DataFrame, List of outlier dates)
    """
    df_copy = df.copy()
    returns = df_copy[column].pct_change()
    z_scores = np.abs((returns - returns.mean()) / returns.std())

    outlier_mask = z_scores > threshold
    outlier_dates = df_copy.index[outlier_mask].tolist()

    # Flag but don't remove (important for historical accuracy)
    df_copy['_is_outlier'] = outlier_mask

    return df_copy, outlier_dates


def validate_data(df: pd.DataFrame, symbol: str) -> dict:
    """
    Validate OHLCV data quality.

    Args:
        df: OHLCV DataFrame
        symbol: Stock symbol

    Returns:
        Dictionary of validation metrics
    """
    validation = {
        'symbol': symbol,
        'total_rows': len(df),
        'date_range': f"{df.index.min().date()} to {df.index.max().date()}",
        'missing_values': df.isnull().sum().to_dict(),
        'zero_volume_days': (df['Volume'] == 0).sum(),
        'duplicate_dates': df.index.duplicated().sum(),
        'ohlc_valid': (df['High'] >= df['Low']).all() and (df['High'] >= df['Close']).all(),
    }

    return validation


def clean_ohlcv(df: pd.DataFrame, symbol: str = 'UNKNOWN') -> pd.DataFrame:
    """
    Comprehensive OHLCV cleaning pipeline.

    Args:
        df: Raw OHLCV DataFrame
        symbol: Stock symbol for validation

    Returns:
        Cleaned DataFrame
    """
    # Step 1: Sort by date
    df = df.sort_index()

    # Step 2: Handle missing prices
    df = forward_fill_prices(df, method='ffill', limit=5)

    # Step 3: Handle outliers (flag but don't remove)
    df, outliers = handle_outliers(df, column='Close', threshold=5.0)

    # Step 4: Validate
    validation = validate_data(df, symbol)

    # Print validation summary
    print(f"\n{symbol} Validation:")
    print(f"  Records: {validation['total_rows']}")
    print(f"  Date Range: {validation['date_range']}")
    print(f"  Zero Volume Days: {validation['zero_volume_days']}")
    print(f"  OHLC Valid: {validation['ohlc_valid']}")
    if len(outliers) > 0:
        print(f"  Outliers Detected: {len(outliers)}")

    return df


def clean_all_data(all_data: dict) -> dict:
    """Clean all stock data."""
    cleaned = {}
    for symbol, df in all_data.items():
        cleaned[symbol] = clean_ohlcv(df, symbol)

    return cleaned


def detect_gaps(df: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
    """
    Detect price gaps (open vs previous close).

    Args:
        df: OHLCV DataFrame
        threshold: Gap threshold (2% = 0.02)

    Returns:
        DataFrame with gap detection
    """
    df_copy = df.copy()
    prev_close = df_copy['Close'].shift(1)
    gap = (df_copy['Open'] - prev_close) / prev_close

    df_copy['Gap'] = gap
    df_copy['_is_gap'] = np.abs(gap) > threshold

    return df_copy


if __name__ == '__main__':
    from fetcher import fetch_nifty_20, load_data_parquet

    # Example: Clean data
    all_data = fetch_nifty_20()
    cleaned = clean_all_data(all_data)

    for symbol, df in cleaned.items():
        df.to_parquet(f"data/processed/{symbol.replace('.NS', '')}_clean.parquet")
