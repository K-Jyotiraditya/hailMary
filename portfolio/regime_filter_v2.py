"""
Tighter Regime Filter v2 — SOTA Continuous Scoring

Implements:
1. ADX-gated regime score: sigmoid(z_score(price/SMA200)) * trend_strength(ADX)
2. Realized Volatility Overlay: Inverse scaling of gross exposure via 20d vol.
"""
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from features.technical import adx

RISK_PROFILES = {
    'HEDGE': {
        'adx_cap': 40.0,
        'vol_target': 0.10,
        'max_leverage': 1.0,
        'sma_window': 200,
    },
    'BALANCED': {
        'adx_cap': 25.0,
        'vol_target': 0.20,
        'max_leverage': 1.5,
        'sma_window': 100,
    },
    'GROWTH': {
        'adx_cap': 15.0, # Relaxes trend requirement significantly
        'vol_target': 0.35, # Aggressively scales up into variance
        'max_leverage': 2.5,
        'sma_window': 50, # Fast moving average
    }
}


def z_score_rolling(series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling Z-score"""
    means = series.rolling(window).mean()
    stds = series.rolling(window).std()
    # Avoid division by zero
    stds = stds.replace(0, 1e-9)
    return (series - means) / stds

def sigmoid(x: pd.Series) -> pd.Series:
    """Stable Sigmoid"""
    return 1 / (1 + np.exp(-x))

def continuous_regime_signal(nifty_df: pd.DataFrame, 
                             sma_window: int = 200,
                             z_window: int = 252,
                             adx_window: int = 14,
                             adx_cap: float = 40.0) -> pd.Series:
    """
    Continuous regime score: sigmoid(z_score(price/SMA200)) * trend_strength(ADX)
    """
    close = nifty_df['Close']
    
    # 1. Price relative to SMA
    sma = close.rolling(sma_window).mean()
    dist_to_sma = close / sma
    
    # 2. Sigmoid of Z-score
    z = z_score_rolling(dist_to_sma, window=z_window).fillna(0)
    base_signal = sigmoid(z)
    
    # 3. ADX Trend Strength
    adx_out = adx(nifty_df['High'], nifty_df['Low'], close, length=adx_window)
    
    # Normalize ADX dynamically. adx_cap adjusts how strict the momentum criteria is
    trend_strength = np.clip(adx_out['adx'] / adx_cap, 0.0, 1.0).fillna(1.0)
    
    # 4. Composite Score
    score = base_signal * trend_strength
    return score.fillna(0.0)

def vol_target_scaler(nifty_df: pd.DataFrame, target_vol_annual: float = 0.15, window: int = 20, max_leverage: float = 1.0) -> pd.Series:
    """
    Returns a daily scaling factor based on index 20-day realized volatility.
    Gross exposure is scaled inversely with trailing 20d vol (capped).
    """
    daily_returns = nifty_df['Close'].pct_change()
    realized_vol_annual = daily_returns.rolling(window).std() * np.sqrt(252)
    # Avoid division by zero
    realized_vol_annual = realized_vol_annual.replace(0, np.nan)
    scaler = target_vol_annual / realized_vol_annual
    return np.clip(scaler, 0.0, max_leverage).fillna(max_leverage)

def composite_bullish_signal(nifty_df: pd.DataFrame, risk_mode: str = 'HEDGE') -> pd.Series:
    """
    Provides the continuous signal combining both regime and volatility.
    Adjusts strictness depending on risk_mode ('HEDGE', 'BALANCED', 'GROWTH').
    """
    prof = RISK_PROFILES.get(risk_mode.upper(), RISK_PROFILES['HEDGE'])
    
    regime = continuous_regime_signal(
        nifty_df, 
        sma_window=prof['sma_window'], 
        adx_cap=prof['adx_cap']
    )
    
    vol_scale = vol_target_scaler(
        nifty_df, 
        target_vol_annual=prof['vol_target'], 
        max_leverage=prof['max_leverage']
    )
    
    # The final scalar weight to scale all positions by
    return regime * vol_scale

if __name__ == '__main__':
    nifty = pd.read_parquet('data/processed/BENCHMARK_INDEX.parquet')
    regime = continuous_regime_signal(nifty)
    vol_scale = vol_target_scaler(nifty)
    final_scale = regime * vol_scale

    print(f"Trading days analyzed: {len(regime.dropna())}")
    print(f"Average Regime Score: {regime.mean():.2f}")
    print(f"Average Vol Scaler:   {vol_scale.mean():.2f}")
    print(f"Average Final Scale:  {final_scale.mean():.2f}")
    
    recent = pd.concat([nifty['Close'], regime, vol_scale, final_scale], axis=1).tail(10)
    recent.columns = ['Close', 'Regime', 'VolScale', 'FinalWeight']
    print("\nRecent Days:")
    print(recent.round(3))
