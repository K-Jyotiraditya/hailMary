"""
Position Sizing — Convert ML signals + HRP weights into portfolio weights.

DESIGN (revised after seeing actual signal strength):
  Our trained models output probabilities mostly in [0.45, 0.60]. A "soft"
  scaling like 2*(p-0.5) gives convictions of [-0.1, 0.2], producing weights
  so small that we'd sit 90% in cash and never compound. That's overcautious
  for the modest-but-real edge we're harvesting.

  We use a HARD-THRESHOLD + HRP approach instead:

    1. A stock is HELD if its predicted probability exceeds `threshold` (default 0.50).
    2. Among held stocks, weights follow the HRP allocation (renormalized so they sum to 1).
    3. Total portfolio exposure scales with average conviction of held stocks:
         exposure = mean( (p - threshold) / (1 - threshold) )    in [0, 1]
       This keeps us in cash when the few stocks above threshold are barely above it.
    4. Weights are capped per-stock to prevent concentration.

  When NO stocks qualify, we hold 100% cash — exactly the defensive behavior
  we want during regime changes or signal blackouts.
"""
import pandas as pd


def compute_target_weights(
    probs: pd.Series,
    hrp_w: pd.Series,
    threshold: float = 0.50,
    max_position: float = 0.40,
    mode: str = 'binary',
) -> pd.Series:
    """
    Compose final long-only target weights from probabilities and HRP weights.

    Args:
        probs: Predicted probabilities per stock for the upcoming period.
        hrp_w: HRP "neutral" weights (sum to 1) over the same universe.
        threshold: Probability above which a stock qualifies for holding.
        max_position: Cap on any single stock's weight.
        mode: 'binary' (full invest in qualifying, 0 if none qualify) or
              'scaled' (exposure scales linearly with conviction).
              Binary works better for weak-signal regimes; scaled fits
              strong, well-calibrated probabilities.

    Returns:
        Series of target weights. Sum is in [0, 1]; remainder is cash.
    """
    aligned = pd.concat([probs, hrp_w], axis=1).dropna()
    aligned.columns = ['prob', 'hrp']

    qualifying = aligned['prob'] > threshold
    if not qualifying.any():
        return pd.Series(0.0, index=probs.index)

    held = aligned[qualifying].copy()

    if mode == 'binary':
        # Fully invested in qualifying stocks, HRP-weighted then renormalized
        held['weight'] = held['hrp'] / held['hrp'].sum()
        # Apply position cap and redistribute the trimmed weight
        held['weight'] = held['weight'].clip(upper=max_position)
        if held['weight'].sum() > 0:
            held['weight'] = held['weight'] / held['weight'].sum()
    elif mode == 'scaled':
        # Soft scaling: weights = HRP-restricted * mean conviction
        held['weight'] = held['hrp'] / held['hrp'].sum()
        conviction = ((held['prob'] - threshold) / (1 - threshold)).clip(0, 1).mean()
        held['weight'] = (held['weight'] * conviction).clip(upper=max_position)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out = pd.Series(0.0, index=probs.index)
    out.loc[held.index] = held['weight'].values
    return out


def turnover(prev_weights: pd.Series, new_weights: pd.Series) -> float:
    """One-way turnover between two weight vectors. Useful for cost estimation."""
    aligned = pd.concat([prev_weights, new_weights], axis=1).fillna(0.0)
    aligned.columns = ['prev', 'new']
    return float((aligned['new'] - aligned['prev']).abs().sum() / 2)


if __name__ == '__main__':
    # Realistic test: most probs near 0.5, some stronger
    probs = pd.Series({
        'AXISBANK': 0.62,
        'ITC':      0.53,
        'NTPC':     0.52,
        'RELIANCE': 0.48,  # below threshold -> excluded
        'SBILIFE':  0.70,
        'TCS':      0.55,
    })
    hrp_w = pd.Series({
        'AXISBANK': 0.135, 'ITC': 0.190, 'NTPC': 0.199,
        'RELIANCE': 0.170, 'SBILIFE': 0.162, 'TCS': 0.143,
    })

    print("=== Strong-signal scenario ===")
    weights = compute_target_weights(probs, hrp_w, threshold=0.50)
    print(weights.round(4).to_string())
    print(f"Total allocated: {weights.sum():.4f}")
    print(f"Cash buffer:     {1 - weights.sum():.4f}\n")

    print("=== Weak-signal scenario (all probs near 0.5) ===")
    weak_probs = pd.Series({k: 0.51 for k in probs.index})
    weak_w = compute_target_weights(weak_probs, hrp_w, threshold=0.50)
    print(weak_w.round(4).to_string())
    print(f"Total allocated: {weak_w.sum():.4f}")
    print(f"Cash buffer:     {1 - weak_w.sum():.4f}\n")

    print("=== Bear-signal scenario (all probs below threshold) ===")
    bear_probs = pd.Series({k: 0.40 for k in probs.index})
    bear_w = compute_target_weights(bear_probs, hrp_w, threshold=0.50)
    print(bear_w.round(4).to_string())
    print(f"Total allocated: {bear_w.sum():.4f}")
    print(f"Cash buffer:     {1 - bear_w.sum():.4f}")
