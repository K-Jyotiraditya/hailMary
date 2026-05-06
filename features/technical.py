"""
Technical Indicators - Custom implementations of standard technical analysis.

All indicators are vectorized for performance and have no external dependencies
beyond pandas/numpy. Each function takes OHLCV data and returns a Series or
DataFrame with the indicator values.
"""
import pandas as pd
import numpy as np


# ============================================================================
# MOMENTUM INDICATORS
# ============================================================================

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index. Range: 0-100. Overbought > 70, oversold < 30."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Wilder's smoothing (EMA with alpha = 1/length)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD: Moving Average Convergence Divergence. Returns line, signal, histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_hist': histogram,
    })


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3, smooth_k: int = 3) -> pd.DataFrame:
    """Stochastic Oscillator. %K and %D. Range 0-100."""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k_raw = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k = k_raw.rolling(smooth_k).mean()
    d = k.rolling(d_period).mean()

    return pd.DataFrame({'stoch_k': k, 'stoch_d': d})


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Williams %R. Range: -100 to 0. Overbought > -20, oversold < -80."""
    highest_high = high.rolling(length).max()
    lowest_low = low.rolling(length).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20) -> pd.Series:
    """Commodity Channel Index. Typically range -100 to +100."""
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(length).mean()
    mean_dev = (typical_price - sma_tp).abs().rolling(length).mean()
    return (typical_price - sma_tp) / (0.015 * mean_dev)


def roc(close: pd.Series, length: int = 10) -> pd.Series:
    """Rate of Change. Percentage change over `length` periods."""
    return 100 * (close - close.shift(length)) / close.shift(length)


# ============================================================================
# TREND INDICATORS
# ============================================================================

def sma(close: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average."""
    return close.rolling(length).mean()


def ema(close: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average."""
    return close.ewm(span=length, adjust=False).mean()


def tema(close: pd.Series, length: int = 10) -> pd.Series:
    """Triple Exponential Moving Average. Reduces lag vs EMA."""
    ema1 = close.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()
    return 3 * ema1 - 3 * ema2 + ema3


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.DataFrame:
    """Average Directional Index. Trend strength (not direction). Range 0-100."""
    tr = true_range(high, low, close)
    atr_val = tr.ewm(alpha=1 / length, adjust=False).mean()

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    plus_di = 100 * plus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_val
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_val
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_val = dx.ewm(alpha=1 / length, adjust=False).mean()

    return pd.DataFrame({'adx': adx_val, 'plus_di': plus_di, 'minus_di': minus_di})


# ============================================================================
# VOLATILITY INDICATORS
# ============================================================================

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range: max(high-low, |high-prev_close|, |low-prev_close|)."""
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range. Wilder's smoothing of True Range."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def bollinger_bands(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: SMA ± k*std. Mean reversion signal."""
    middle = close.rolling(length).mean()
    sd = close.rolling(length).std()
    upper = middle + std * sd
    lower = middle - std * sd
    width = (upper - lower) / middle
    pct_b = (close - lower) / (upper - lower)

    return pd.DataFrame({
        'bb_middle': middle,
        'bb_upper': upper,
        'bb_lower': lower,
        'bb_width': width,
        'bb_pct': pct_b,
    })


def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                     length: int = 20, atr_mult: float = 2.0) -> pd.DataFrame:
    """Keltner Channels: EMA ± k*ATR."""
    middle = close.ewm(span=length, adjust=False).mean()
    atr_val = atr(high, low, close, length)
    upper = middle + atr_mult * atr_val
    lower = middle - atr_mult * atr_val

    return pd.DataFrame({
        'kc_middle': middle,
        'kc_upper': upper,
        'kc_lower': lower,
    })


# ============================================================================
# VOLUME INDICATORS
# ============================================================================

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume. Cumulative volume signed by price direction."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume-Weighted Average Price (cumulative since start of series)."""
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()


def volume_roc(volume: pd.Series, length: int = 10) -> pd.Series:
    """Volume Rate of Change."""
    return 100 * (volume - volume.shift(length)) / volume.shift(length).replace(0, np.nan)


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 14) -> pd.Series:
    """Money Flow Index. Volume-weighted RSI. Range 0-100."""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    pos_sum = positive_flow.rolling(length).sum()
    neg_sum = negative_flow.rolling(length).sum()

    money_ratio = pos_sum / neg_sum
    return 100 - (100 / (1 + money_ratio))


# ============================================================================
# MASTER FUNCTION
# ============================================================================

def compute_all_technical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators for an OHLCV DataFrame.

    Returns a DataFrame with all indicator columns, indexed by the same dates.
    """
    out = pd.DataFrame(index=df.index)

    # Momentum
    out['rsi_14'] = rsi(df['Close'], 14)
    out['rsi_28'] = rsi(df['Close'], 28)
    macd_df = macd(df['Close'], 12, 26, 9)
    out = pd.concat([out, macd_df], axis=1)
    stoch_df = stochastic(df['High'], df['Low'], df['Close'], 14, 3, 3)
    out = pd.concat([out, stoch_df], axis=1)
    out['williams_r'] = williams_r(df['High'], df['Low'], df['Close'], 14)
    out['cci_20'] = cci(df['High'], df['Low'], df['Close'], 20)
    out['roc_10'] = roc(df['Close'], 10)
    out['roc_20'] = roc(df['Close'], 20)

    # Trend
    out['sma_20'] = sma(df['Close'], 20)
    out['sma_50'] = sma(df['Close'], 50)
    out['sma_200'] = sma(df['Close'], 200)
    out['ema_12'] = ema(df['Close'], 12)
    out['ema_26'] = ema(df['Close'], 26)
    out['tema_10'] = tema(df['Close'], 10)
    adx_df = adx(df['High'], df['Low'], df['Close'], 14)
    out = pd.concat([out, adx_df], axis=1)

    # Trend ratios (price vs MA)
    out['close_vs_sma20'] = df['Close'] / out['sma_20'] - 1
    out['close_vs_sma50'] = df['Close'] / out['sma_50'] - 1
    out['close_vs_sma200'] = df['Close'] / out['sma_200'] - 1
    out['sma20_vs_sma50'] = out['sma_20'] / out['sma_50'] - 1

    # Volatility
    out['atr_14'] = atr(df['High'], df['Low'], df['Close'], 14)
    out['atr_pct'] = out['atr_14'] / df['Close']  # Normalized ATR
    bb_df = bollinger_bands(df['Close'], 20, 2.0)
    out = pd.concat([out, bb_df], axis=1)
    kc_df = keltner_channels(df['High'], df['Low'], df['Close'], 20, 2.0)
    out = pd.concat([out, kc_df], axis=1)

    # Volume
    out['obv'] = obv(df['Close'], df['Volume'])
    out['vwap'] = vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    out['close_vs_vwap'] = df['Close'] / out['vwap'] - 1
    out['volume_roc_10'] = volume_roc(df['Volume'], 10)
    out['mfi_14'] = mfi(df['High'], df['Low'], df['Close'], df['Volume'], 14)

    return out


if __name__ == '__main__':
    df = pd.read_parquet('data/processed/RELIANCE.parquet')
    features = compute_all_technical(df)
    print(f"Computed {features.shape[1]} technical features for {features.shape[0]} dates")
    print(f"\nFeature columns:")
    for col in features.columns:
        print(f"  {col}")
    print(f"\nLast row (most recent):")
    print(features.iloc[-1].round(4))
