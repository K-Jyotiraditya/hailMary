"""
Price-Based Features - Returns, volatility, skewness, kurtosis, gaps.

These complement technical indicators by capturing distributional and
microstructure properties of price action.
"""
import pandas as pd
import numpy as np


def log_returns(close: pd.Series, lag: int = 1) -> pd.Series:
    """Log returns over `lag` periods. log(P_t / P_{t-lag})."""
    return np.log(close / close.shift(lag))


def simple_returns(close: pd.Series, lag: int = 1) -> pd.Series:
    """Simple percentage returns."""
    return close.pct_change(lag)


def rolling_volatility(returns: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """Rolling standard deviation of returns. Annualized by default (252 days)."""
    vol = returns.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def rolling_skewness(returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling skewness. Negative = left tail risk, positive = right tail."""
    return returns.rolling(window).skew()


def rolling_kurtosis(returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling kurtosis (excess). High values = fat tails."""
    return returns.rolling(window).kurt()


def gap(open_: pd.Series, prev_close: pd.Series) -> pd.Series:
    """Overnight gap as percentage of previous close."""
    return (open_ - prev_close) / prev_close


def high_low_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Daily range as percentage of close."""
    return (high - low) / close


def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Parkinson volatility estimator using high-low range.
    More efficient than close-to-close for noisy data.
    """
    log_hl = np.log(high / low)
    var = (log_hl ** 2 / (4 * np.log(2))).rolling(window).mean()
    return np.sqrt(var * 252)  # Annualized


def garman_klass_volatility(open_: pd.Series, high: pd.Series, low: pd.Series,
                            close: pd.Series, window: int = 20) -> pd.Series:
    """
    Garman-Klass volatility estimator. Uses OHLC.
    More efficient than Parkinson when data has open and close.
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    var_per_day = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    var = var_per_day.rolling(window).mean()
    return np.sqrt(var * 252)


def downside_volatility(returns: pd.Series, window: int = 20, threshold: float = 0.0) -> pd.Series:
    """
    Downside volatility (semi-deviation). Only counts negative deviations.
    Used for Sortino ratio.
    """
    downside = returns.where(returns < threshold, 0.0)
    return downside.rolling(window).std() * np.sqrt(252)


def max_drawdown_rolling(close: pd.Series, window: int = 252) -> pd.Series:
    """Rolling maximum drawdown over `window` days."""
    rolling_max = close.rolling(window, min_periods=1).max()
    drawdown = (close - rolling_max) / rolling_max
    return drawdown.rolling(window, min_periods=1).min()


def autocorrelation(returns: pd.Series, lag: int = 1, window: int = 60) -> pd.Series:
    """Rolling autocorrelation of returns at given lag."""
    return returns.rolling(window).apply(
        lambda x: x.autocorr(lag=lag) if len(x.dropna()) > lag else np.nan,
        raw=False,
    )


def compute_all_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all price-based features for an OHLCV DataFrame.
    """
    out = pd.DataFrame(index=df.index)

    # Returns at various horizons
    out['log_ret_1'] = log_returns(df['Close'], 1)
    out['log_ret_5'] = log_returns(df['Close'], 5)
    out['log_ret_20'] = log_returns(df['Close'], 20)

    # Cumulative returns (multi-day momentum)
    out['ret_5d'] = simple_returns(df['Close'], 5)
    out['ret_20d'] = simple_returns(df['Close'], 20)
    out['ret_60d'] = simple_returns(df['Close'], 60)

    # Volatility (multiple windows)
    daily_ret = simple_returns(df['Close'], 1)
    out['vol_20'] = rolling_volatility(daily_ret, 20, annualize=True)
    out['vol_60'] = rolling_volatility(daily_ret, 60, annualize=True)
    out['vol_ratio_20_60'] = out['vol_20'] / out['vol_60']  # Vol regime

    # Higher moments
    out['skew_20'] = rolling_skewness(daily_ret, 20)
    out['skew_60'] = rolling_skewness(daily_ret, 60)
    out['kurt_20'] = rolling_kurtosis(daily_ret, 20)

    # Range-based volatility (more efficient estimators)
    out['vol_parkinson_20'] = parkinson_volatility(df['High'], df['Low'], 20)
    out['vol_gk_20'] = garman_klass_volatility(df['Open'], df['High'], df['Low'], df['Close'], 20)

    # Downside risk
    out['downside_vol_20'] = downside_volatility(daily_ret, 20)

    # Microstructure features
    out['gap'] = gap(df['Open'], df['Close'].shift(1))
    out['hl_range'] = high_low_range(df['High'], df['Low'], df['Close'])
    out['hl_range_5d_avg'] = out['hl_range'].rolling(5).mean()

    # Drawdown
    out['drawdown_60'] = max_drawdown_rolling(df['Close'], 60)
    out['drawdown_252'] = max_drawdown_rolling(df['Close'], 252)

    # Autocorrelation (mean reversion vs momentum signal)
    out['autocorr_1'] = autocorrelation(daily_ret, lag=1, window=60)
    out['autocorr_5'] = autocorrelation(daily_ret, lag=5, window=60)

    # Volume features
    out['volume_ma_20'] = df['Volume'].rolling(20).mean()
    out['volume_ratio'] = df['Volume'] / out['volume_ma_20']
    out['log_volume'] = np.log(df['Volume'].replace(0, np.nan))

    # Dollar volume (liquidity proxy)
    out['dollar_volume_ma_20'] = (df['Close'] * df['Volume']).rolling(20).mean()

    return out


if __name__ == '__main__':
    df = pd.read_parquet('data/processed/RELIANCE.parquet')
    features = compute_all_price_features(df)
    print(f"Computed {features.shape[1]} price-based features for {features.shape[0]} dates")
    print(f"\nFeature columns:")
    for col in features.columns:
        print(f"  {col}")
    print(f"\nLast row (most recent):")
    print(features.iloc[-1].round(4))
