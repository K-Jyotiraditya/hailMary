"""
Top-K Selection — pick the K best stocks by predicted rank, weight with HRP.

Different from Phase 1's threshold approach:
  Phase 1: hold a fixed universe (6 stocks) when each stock's prob > 0.5.
  Phase 2: pick the TOP K stocks across NIFTY 46 each rebalance, regardless of
           absolute score. This naturally filters and ranks at the same time.

Why top-K is better with cross-sectional rank predictions:
  - Always has positions if there ARE predictions (most rebalances)
  - Adapts to market — picks the relatively strongest, even when all are weak
  - Pairs naturally with HRP for risk-balanced sizing within the K winners
  - Built-in turnover control (only changes when ranks change)
"""
import pandas as pd
import numpy as np


def select_top_k(predictions: pd.Series, k: int = 10) -> list:
    """
    Pick the top K symbols by predicted rank.

    Args:
        predictions: Series indexed by symbol, values = predicted rank.
        k: Number of stocks to select.

    Returns:
        List of selected symbol names (ordered by descending prediction).
    """
    return predictions.nlargest(k).index.tolist()


def top_k_with_hrp(
    predictions: pd.Series,
    returns_window: pd.DataFrame,
    k: int = 10,
    max_position: float = 0.15,
    min_predictions: int = 5,
) -> pd.Series:
    """
    Select top-K stocks by prediction and HRP-weight them.

    Args:
        predictions: Predicted rank per stock for the upcoming period.
        returns_window: DataFrame of past returns over the cov-window for HRP.
        k: Number of stocks to hold.
        max_position: Cap on any single stock weight (concentration limit).
        min_predictions: Minimum non-NaN predictions to operate (else cash).

    Returns:
        Series of weights summing to <= 1; remainder is cash.
    """
    from portfolio.optimizer import hrp_weights_robust

    valid = predictions.dropna()
    if len(valid) < min_predictions:
        return pd.Series(0.0, index=predictions.index)

    selected = select_top_k(valid, k=min(k, len(valid)))
    if not selected:
        return pd.Series(0.0, index=predictions.index)

    # HRP among the selected stocks
    sel_returns = returns_window[selected].dropna(how='any')
    if len(sel_returns) >= 30:
        hrp_w = hrp_weights_robust(sel_returns)
    else:
        hrp_w = pd.Series(1.0 / len(selected), index=selected)

    # Apply concentration cap and renormalize
    hrp_w = hrp_w.clip(upper=max_position)
    if hrp_w.sum() > 0:
        hrp_w = hrp_w / hrp_w.sum()

    # Build full weight vector across the full universe
    out = pd.Series(0.0, index=predictions.index)
    out.loc[selected] = hrp_w.values
    return out


def top_k_equal_weight(predictions: pd.Series, k: int = 10) -> pd.Series:
    """Simpler alternative: top K equally weighted (1/K each)."""
    valid = predictions.dropna()
    if len(valid) < 1:
        return pd.Series(0.0, index=predictions.index)
    selected = select_top_k(valid, k=min(k, len(valid)))
    out = pd.Series(0.0, index=predictions.index)
    if selected:
        out.loc[selected] = 1.0 / len(selected)
    return out


if __name__ == '__main__':
    # Smoke test
    preds = pd.Series({
        'STK_A': 0.85, 'STK_B': 0.75, 'STK_C': 0.20, 'STK_D': 0.95, 'STK_E': 0.40,
        'STK_F': 0.60, 'STK_G': 0.30, 'STK_H': 0.55, 'STK_I': 0.70, 'STK_J': 0.10,
    })
    print("Top-5 by prediction:", select_top_k(preds, k=5))
    print("\nTop-3 equal-weight:")
    print(top_k_equal_weight(preds, k=3).round(4))
