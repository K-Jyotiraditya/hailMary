"""
Rebalancing Schedule — Decide WHEN to rebalance the portfolio.

Per the Phase 1 plan, we rebalance monthly on the first trading day of the month.
That balances signal recency against transaction costs: with our IC of ~0.02
and 5 bps round-trip cost, weekly rebalancing would eat too much of the edge.
"""
import pandas as pd


def monthly_rebalance_dates(trading_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Pick the first trading date of each month from a DatetimeIndex.
    """
    df = pd.DataFrame(index=trading_dates)
    df['year_month'] = df.index.to_period('M')
    first_per_month = df.groupby('year_month').apply(lambda g: g.index[0])
    return pd.DatetimeIndex(first_per_month.values)


def weekly_rebalance_dates(trading_dates: pd.DatetimeIndex,
                           weekday: int = 0) -> pd.DatetimeIndex:
    """First trading day of each week on/after `weekday` (0=Monday)."""
    df = pd.DataFrame(index=trading_dates)
    df['week'] = df.index.to_period('W')
    first_per_week = df.groupby('week').apply(lambda g: g.index[0])
    return pd.DatetimeIndex(first_per_week.values)


def quarterly_rebalance_dates(trading_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First trading day of each quarter."""
    df = pd.DataFrame(index=trading_dates)
    df['quarter'] = df.index.to_period('Q')
    first_per_q = df.groupby('quarter').apply(lambda g: g.index[0])
    return pd.DatetimeIndex(first_per_q.values)


def get_rebalance_dates(trading_dates: pd.DatetimeIndex,
                        frequency: str = 'monthly') -> pd.DatetimeIndex:
    """Dispatch on frequency string."""
    if frequency == 'monthly':
        return monthly_rebalance_dates(trading_dates)
    if frequency == 'weekly':
        return weekly_rebalance_dates(trading_dates)
    if frequency == 'quarterly':
        return quarterly_rebalance_dates(trading_dates)
    if frequency == 'daily':
        return trading_dates
    raise ValueError(f"Unknown frequency: {frequency}")


if __name__ == '__main__':
    trading = pd.bdate_range('2023-01-01', '2024-12-31')
    monthly = get_rebalance_dates(trading, 'monthly')
    print(f"Monthly rebalance dates ({len(monthly)} total):")
    print(monthly[:6].tolist())
    print(f"\nQuarterly: {len(get_rebalance_dates(trading, 'quarterly'))}")
    print(f"Weekly:    {len(get_rebalance_dates(trading, 'weekly'))}")
