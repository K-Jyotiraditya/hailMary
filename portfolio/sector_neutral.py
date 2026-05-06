"""
Sector-Neutral Top-K Selection — prevent any sector from dominating.

The default top-K + HRP selects stocks purely on predicted rank. With a
universe heavy in IT and Banking, the top-10 often ends up 70%+ in those
two sectors. That's a hidden bet on sector returns, not stock-specific
alpha.

This module forces a per-sector cap: no single sector contributes more than
`max_sector_weight` (default 35%) of total invested capital. We achieve this
by:

  1. Pick top-K candidates by prediction.
  2. Compute their HRP weights.
  3. If any sector exceeds the cap, scale down ALL stocks in that sector
     proportionally and redistribute the trimmed weight to under-allocated
     sectors (also proportionally to their existing weights).
  4. Iterate until all caps are satisfied (usually 1-2 passes).

This keeps the portfolio's sector exposure close to whatever HRP would have
chosen, but enforces a hard ceiling on concentration risk.
"""
import pandas as pd
import numpy as np

from data.sectors import get_sector, sector_distribution


def enforce_sector_cap(weights: pd.Series, max_sector_weight: float = 0.35,
                       max_position: float = 0.18,
                       max_iterations: int = 20) -> pd.Series:
    """
    Cap each sector's total weight AND each stock's weight. Excess that can't
    be redistributed (because all under-allocated stocks already hit per-stock
    cap) becomes cash — the safe default rather than violating either cap.

    Args:
        weights: Series indexed by symbol, values = portfolio weights.
        max_sector_weight: Hard cap per sector.
        max_position: Hard cap per single stock.
        max_iterations: Safety bound; usually converges fast.

    Returns:
        Capped weights. Total may be < original if both caps couldn't be
        simultaneously satisfied — the difference is cash.
    """
    w = weights.copy().astype(float)
    if w.sum() <= 0:
        return w

    sector_of = {s: get_sector(s) for s in w.index}
    sectors = sorted(set(sector_of.values()))
    eps = 1e-6

    for _ in range(max_iterations):
        total = float(w.sum())
        if total <= eps:
            break

        # Compute per-sector totals
        sec_totals = {sec: 0.0 for sec in sectors}
        for s, ww in w.items():
            sec_totals[sector_of[s]] += float(ww)

        # 1) Cap individual stocks
        stock_overshoot = (w - max_position).clip(lower=0).sum()
        w = w.clip(upper=max_position)

        # 2) Cap each sector
        sector_excess = 0.0
        for sec, sec_total in sec_totals.items():
            if sec_total <= max_sector_weight + eps:
                continue
            sector_stocks = [s for s in w.index if sector_of[s] == sec]
            target_total = max_sector_weight  # absolute, since we're treating as fractions of 1.0
            cur = sum(w[s] for s in sector_stocks)
            if cur <= 0 or cur <= target_total:
                continue
            scale = target_total / cur
            for s in sector_stocks:
                sector_excess += w[s] * (1 - scale)
                w[s] *= scale

        excess = stock_overshoot + sector_excess
        if excess <= eps:
            # Re-check: are all caps now satisfied?
            sec_totals_now = {sec: 0.0 for sec in sectors}
            for s, ww in w.items():
                sec_totals_now[sector_of[s]] += float(ww)
            stock_ok = (w <= max_position + eps).all()
            sector_ok = all(t <= max_sector_weight + eps for t in sec_totals_now.values())
            if stock_ok and sector_ok:
                break
            # Otherwise loop again to re-trim

        # Redistribute excess to stocks that have HEADROOM under both caps
        # Headroom for a stock = min(max_position - w[s], max_sector_weight - sector_total)
        sec_totals_now = {sec: 0.0 for sec in sectors}
        for s, ww in w.items():
            sec_totals_now[sector_of[s]] += float(ww)

        headroom = {}
        for s in w.index:
            stock_room = max_position - w[s]
            sec_room = max_sector_weight - sec_totals_now[sector_of[s]]
            room = min(max(stock_room, 0.0), max(sec_room, 0.0))
            if room > eps:
                headroom[s] = room

        if not headroom:
            # No room anywhere — leave excess as cash, exit
            break

        # Distribute proportionally to existing weight where there's headroom
        existing = pd.Series({s: w[s] for s in headroom})
        if existing.sum() <= 0:
            existing = pd.Series({s: 1.0 for s in headroom})

        share = existing / existing.sum()
        added = 0.0
        for s in headroom:
            add = min(excess * share[s], headroom[s])
            w[s] += add
            added += add
        if added <= eps:
            break

    return w


def sector_neutral_top_k(
    predictions: pd.Series,
    returns_window: pd.DataFrame,
    k: int = 10,
    max_position: float = 0.18,
    max_sector_weight: float = 0.35,
    min_sectors: int = 4,
) -> pd.Series:
    """
    Top-K selection with sector neutrality.

    First pass picks top-K by prediction. If those K cover < min_sectors
    different sectors, we expand the candidate pool until we get diversity.
    Then we apply HRP and the sector cap.

    Args:
        predictions: Predicted ranks per stock (Series).
        returns_window: Past returns for HRP covariance (DataFrame).
        k: Initial number of top stocks to consider.
        max_position: Single-stock weight cap.
        max_sector_weight: Per-sector total weight cap.
        min_sectors: Minimum unique sectors represented in the portfolio.
    """
    from portfolio.optimizer import hrp_weights_robust

    valid = predictions.dropna()
    if len(valid) < k:
        return pd.Series(0.0, index=predictions.index)

    # Pick initial top-K
    sorted_preds = valid.sort_values(ascending=False)
    top = sorted_preds.head(k)
    sectors_covered = set(get_sector(s) for s in top.index)

    # Expand candidate pool until min_sectors are represented
    extra_pool = sorted_preds.iloc[k:]
    for sym in extra_pool.index:
        if len(sectors_covered) >= min_sectors:
            break
        sec = get_sector(sym)
        if sec not in sectors_covered:
            top = pd.concat([top, sorted_preds.loc[[sym]]])
            sectors_covered.add(sec)

    selected = top.index.tolist()

    # HRP weights within selection
    sel_returns = returns_window[selected].dropna(how='any')
    if len(sel_returns) >= 30:
        try:
            hrp_w = hrp_weights_robust(sel_returns)
        except Exception:
            hrp_w = pd.Series(1.0 / len(selected), index=selected)
    else:
        hrp_w = pd.Series(1.0 / len(selected), index=selected)

    # Single-stock cap, then renormalize
    hrp_w = hrp_w.clip(upper=max_position)
    if hrp_w.sum() > 0:
        hrp_w = hrp_w / hrp_w.sum()

    # Sector cap
    hrp_w = enforce_sector_cap(hrp_w, max_sector_weight=max_sector_weight)

    out = pd.Series(0.0, index=predictions.index)
    out.loc[selected] = hrp_w.values
    return out


if __name__ == '__main__':
    # Smoke test: heavy IT/banking portfolio gets reshaped
    weights = pd.Series({
        'WIPRO': 0.12, 'HDFCBANK': 0.11, 'INFY': 0.10, 'HDFCLIFE': 0.11,
        'TCS': 0.10, 'ICICIBANK': 0.12, 'HCLTECH': 0.09, 'BAJAJFINSV': 0.10,
        'BPCL': 0.08, 'KOTAKBANK': 0.07,
    })
    print("Original distribution:")
    print(sector_distribution(weights))

    capped = enforce_sector_cap(weights, max_sector_weight=0.35)
    print("\nAfter 35% sector cap:")
    print(sector_distribution(capped))
    print("\nNew weights:")
    print(capped.round(4).to_string())
    print(f"\nTotal: {capped.sum():.4f} (should be 1.0)")
