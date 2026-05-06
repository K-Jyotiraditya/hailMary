"""
Vol-Targeted Position Sizing — keep portfolio realized volatility near a target.

Why this matters:
  Without vol-targeting, position sizing is governed by HRP cluster weights,
  which adapt to RELATIVE risk between stocks but ignore the GLOBAL level of
  risk. In high-vol regimes (e.g., 2020-Q1), a normally-sized portfolio has
  much more dollar volatility than usual. Vol-targeting fixes that: scale
  total exposure down when realized vol is high, up when it's low.

  This is how almost every institutional risk-parity / managed-futures fund
  controls drawdowns. It's a one-line addition that meaningfully improves
  Sharpe and reduces tail risk.

Algorithm:
  1. Compute portfolio realized vol over a lookback (e.g., 20-60 days) from
     the per-stock returns and current weights.
  2. Compute target vol leverage = TARGET / current_vol.
  3. Scale ALL weights by the leverage ratio, capped to avoid extreme leverage.
"""
import numpy as np
import pandas as pd


def realized_portfolio_vol(weights: pd.Series, returns: pd.DataFrame,
                           window: int = 20) -> float:
    """
    Compute the trailing portfolio volatility implied by current weights and
    a rolling-window covariance of stock returns. Annualized (sqrt(252)).
    """
    # Use only stocks with actual weight
    held = weights[weights.abs() > 1e-6]
    if held.empty:
        return 0.0

    relevant = returns[held.index].dropna(how='any').tail(window)
    if len(relevant) < window // 2:
        return 0.0

    cov = relevant.cov()
    w = held.reindex(cov.index).fillna(0.0)
    portfolio_var = float(w @ cov.values @ w.values)
    return np.sqrt(max(portfolio_var, 0)) * np.sqrt(252)


def vol_target_scale(weights: pd.Series, returns: pd.DataFrame,
                     target_vol: float = 0.12, window: int = 20,
                     min_leverage: float = 0.0,
                     max_leverage: float = 1.0) -> pd.Series:
    """
    Scale weights so realized portfolio vol matches `target_vol`.

    Args:
        weights: current target weights (sum <= 1).
        returns: daily returns DataFrame for the universe.
        target_vol: desired annualized portfolio volatility (e.g., 0.12 = 12%).
        window: lookback window for realized vol (days).
        min_leverage: floor on the scale factor (don't go fully to cash on low vol).
        max_leverage: cap on the scale factor (no leverage by default).

    Returns:
        Scaled weights. Sum may be < 1 (cash buffer) — kept long-only.
    """
    if weights.abs().sum() < 1e-6:
        return weights

    current_vol = realized_portfolio_vol(weights, returns, window=window)
    if current_vol <= 0:
        return weights

    leverage = target_vol / current_vol
    leverage = np.clip(leverage, min_leverage, max_leverage)
    return weights * leverage


def vol_target_with_floor(weights: pd.Series, returns: pd.DataFrame,
                          target_vol: float = 0.12, window: int = 20,
                          min_exposure: float = 0.20,
                          max_leverage: float = 1.0) -> pd.Series:
    """
    Vol-target while keeping at least `min_exposure` invested when signals exist.

    Useful when you don't want the strategy to go to nearly-zero exposure during
    extreme vol events — keep some skin in the game.
    """
    scaled = vol_target_scale(weights, returns, target_vol=target_vol,
                              window=window, min_leverage=0.0,
                              max_leverage=max_leverage)
    # If scaled total exposure dropped below min_exposure, lift back up
    if scaled.abs().sum() > 0 and scaled.abs().sum() < min_exposure:
        scaled = scaled / scaled.abs().sum() * min_exposure
    return scaled


if __name__ == '__main__':
    # Smoke test
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=60, freq='B')
    rets = pd.DataFrame(
        np.random.randn(60, 4) * 0.015,  # 1.5% daily vol per stock
        index=dates, columns=['A', 'B', 'C', 'D'],
    )
    w = pd.Series([0.25, 0.25, 0.25, 0.25], index=['A', 'B', 'C', 'D'])
    realized = realized_portfolio_vol(w, rets, window=20)
    print(f"Realized portfolio vol: {realized:.2%} (annualized)")
    scaled = vol_target_scale(w, rets, target_vol=0.10, window=20)
    print(f"Original weights:   {w.values}")
    print(f"Vol-targeted (10%): {scaled.values}")
    print(f"New realized vol:   {realized_portfolio_vol(scaled, rets, 20):.2%}")
