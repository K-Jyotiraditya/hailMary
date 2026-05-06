"""
Triple-Barrier Labeling (de Prado's method, AFML Chapter 3).

For each entry timestamp t, we set three barriers:
  - Upper (profit target): price moves up by `pt_mult * sigma_t`
  - Lower (stop loss):     price moves down by `sl_mult * sigma_t`
  - Vertical (time):       max_holding bars after entry

The label is determined by which barrier is hit FIRST:
  +1 if upper hit first (profit)
  -1 if lower hit first (stop loss)
   0 if time barrier hit first (or sign of return for trinary version)

WHY THIS IS BETTER THAN FIXED RETURNS:
  Fixed % target (e.g., "did the stock return >1% in 5 days?") ignores
  volatility regime. A 1% move in TCS during calm market is significant;
  a 1% move during chaos is noise. Volatility-scaled barriers solve this
  by adapting to recent realized volatility.

WHY IT'S BETTER THAN BINARY UP/DOWN:
  Binary "up vs down" labels ignore path dependency. A stock that grinds
  up 0.5% over 5 days is treated the same as one that whipsaws ±5% but
  closes +0.5%. Triple-barrier captures the actual trading experience.
"""
import pandas as pd
import numpy as np


def daily_volatility(close: pd.Series, span: int = 20) -> pd.Series:
    """
    Compute exponentially-weighted daily volatility from log returns.
    This is the basis for sizing the triple barriers.
    """
    log_ret = np.log(close / close.shift(1))
    return log_ret.ewm(span=span).std()


def triple_barrier_labels(
    close: pd.Series,
    events: pd.DatetimeIndex = None,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    max_holding: int = 5,
    vol_span: int = 20,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """
    Apply the triple-barrier method to a price series.

    Args:
        close: Price series (typically Close prices).
        events: Subset of timestamps at which to apply labeling.
                If None, every date in `close` is an event.
        pt_mult: Profit target as multiple of daily volatility.
                 e.g., pt_mult=2.0 means target is 2*sigma above entry.
        sl_mult: Stop loss as multiple of daily volatility.
        max_holding: Maximum holding period in bars (trading days).
        vol_span: Span for EWM volatility estimation.
        min_ret: Minimum return threshold to register a label
                 (filters out micro-moves that hit barriers by accident).

    Returns:
        DataFrame indexed by event timestamps with columns:
            t1: timestamp when label was determined (barrier hit time)
            ret: signed return from entry to barrier hit
            bin: label in {-1, 0, +1}
            barrier: which barrier was hit ('pt', 'sl', 'time')
            pt_price: profit-target price level
            sl_price: stop-loss price level
            holding_days: number of bars held
    """
    if events is None:
        events = close.index

    # Compute daily volatility
    vol = daily_volatility(close, span=vol_span)

    # Pre-compute index positions for fast lookups
    idx_positions = {ts: i for i, ts in enumerate(close.index)}

    results = []

    for entry_ts in events:
        if entry_ts not in idx_positions:
            continue

        i = idx_positions[entry_ts]
        if i + 1 >= len(close):
            continue  # No future data

        entry_price = close.iloc[i]
        sigma = vol.iloc[i]

        if pd.isna(sigma) or sigma <= 0:
            continue  # Skip if volatility undefined

        # Set barriers (in price terms)
        pt_price = entry_price * np.exp(pt_mult * sigma)
        sl_price = entry_price * np.exp(-sl_mult * sigma)

        # Time horizon: up to max_holding bars ahead
        end_i = min(i + max_holding, len(close) - 1)
        future_path = close.iloc[i + 1: end_i + 1]

        # Detect barrier hits
        hit_pt = future_path >= pt_price
        hit_sl = future_path <= sl_price

        first_pt = hit_pt.idxmax() if hit_pt.any() else None
        first_sl = hit_sl.idxmax() if hit_sl.any() else None

        # Determine which barrier hits first
        if first_pt is not None and (first_sl is None or first_pt <= first_sl):
            barrier = 'pt'
            t1 = first_pt
            exit_price = close.loc[t1]
            label = 1
        elif first_sl is not None:
            barrier = 'sl'
            t1 = first_sl
            exit_price = close.loc[t1]
            label = -1
        else:
            # Time barrier
            barrier = 'time'
            t1 = future_path.index[-1] if len(future_path) > 0 else entry_ts
            exit_price = close.loc[t1]
            ret = exit_price / entry_price - 1
            if abs(ret) < min_ret:
                label = 0
            else:
                label = int(np.sign(ret))

        ret = exit_price / entry_price - 1
        holding = (close.index.get_loc(t1) - i) if t1 != entry_ts else 0

        results.append({
            'entry_ts': entry_ts,
            't1': t1,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'ret': ret,
            'bin': label,
            'barrier': barrier,
            'pt_price': pt_price,
            'sl_price': sl_price,
            'holding_days': holding,
            'sigma': sigma,
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).set_index('entry_ts')
    df.index.name = 'date'
    return df


def get_label_summary(labels: pd.DataFrame) -> dict:
    """Summary statistics for a labeled dataset."""
    if labels.empty:
        return {}

    total = len(labels)
    pt_count = (labels['barrier'] == 'pt').sum()
    sl_count = (labels['barrier'] == 'sl').sum()
    time_count = (labels['barrier'] == 'time').sum()

    return {
        'n_labels': total,
        'pct_profit_target': pt_count / total,
        'pct_stop_loss': sl_count / total,
        'pct_time_barrier': time_count / total,
        'class_balance': labels['bin'].value_counts(normalize=True).to_dict(),
        'mean_holding_days': labels['holding_days'].mean(),
        'mean_return': labels['ret'].mean(),
        'win_rate': (labels['bin'] == 1).sum() / total,
    }


if __name__ == '__main__':
    df = pd.read_parquet('data/processed/RELIANCE.parquet')

    # Apply triple-barrier
    labels = triple_barrier_labels(
        df['Close'],
        pt_mult=2.0,
        sl_mult=2.0,
        max_holding=5,
        vol_span=20,
    )

    summary = get_label_summary(labels)
    print(f"\nTRIPLE-BARRIER LABELING — RELIANCE")
    print(f"=" * 60)
    print(f"Total labeled events:   {summary['n_labels']}")
    print(f"Hit profit target (PT): {summary['pct_profit_target']:.1%}")
    print(f"Hit stop loss (SL):     {summary['pct_stop_loss']:.1%}")
    print(f"Hit time barrier:       {summary['pct_time_barrier']:.1%}")
    print(f"\nClass balance:")
    for cls, pct in sorted(summary['class_balance'].items()):
        print(f"  {cls:+d}: {pct:.1%}")
    print(f"\nMean holding days:      {summary['mean_holding_days']:.2f}")
    print(f"Mean return per trade:  {summary['mean_return']:.4f}")
    print(f"Win rate:               {summary['win_rate']:.1%}")
    print(f"\nSample labels:")
    print(labels[['ret', 'bin', 'barrier', 'holding_days']].tail(5))
