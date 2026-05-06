"""
Vectorized Backtest Engine.

DESIGN:
  Discrete-event backtest with daily mark-to-market. Rebalances happen on
  pre-specified dates (default monthly). On each rebalance:
    1. Compute target weights from a signal function.
    2. Calculate trades (delta vs current portfolio).
    3. Apply slippage on the trade side AND commission on traded notional.
    4. Update positions and cash.
  Between rebalances, positions drift with prices; cash earns nothing.

  Critical realism details:
    - Trades execute at next-bar's CLOSE (not the rebalance signal date),
      so signals computed on day t are tradable on day t+1. This avoids
      look-ahead in the backtest itself.
    - Slippage applied on the WORSE side (buy higher, sell lower).
    - Commission charged on absolute traded value.
"""
import numpy as np
import pandas as pd
from typing import Callable
from dataclasses import dataclass, field


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    slippage_bps: float = 1.0       # 1 bp = 0.01%
    commission_bps: float = 5.0     # 5 bps = 0.05% per trade
    cash_return: float = 0.0        # Annualized risk-free; default 0


@dataclass
class BacktestResults:
    equity_curve: pd.Series
    positions: pd.DataFrame              # daily share holdings
    weights: pd.DataFrame                # daily weights
    cash: pd.Series
    trades: pd.DataFrame                 # rebalance-day trades (rows = rebalance dates)
    turnover: pd.Series                  # one-way turnover per rebalance
    returns: pd.Series = field(init=False)

    def __post_init__(self):
        self.returns = self.equity_curve.pct_change().dropna()


def run_backtest(
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    signal_fn: Callable[[pd.Timestamp], pd.Series],
    config: BacktestConfig = None,
    volumes: pd.DataFrame = None,
) -> BacktestResults:
    """
    Run the backtest.

    Args:
        prices: DataFrame with columns = stocks, rows = trading days, values = Close.
        rebalance_dates: Dates on which signals are computed.
        signal_fn: Callable f(date) -> Series of TARGET WEIGHTS for the
                   universe (sum <= 1). Date is the SIGNAL date; trades
                   execute at the next trading day's close.
        config: Backtest parameters.
        volumes: Optional DataFrame with same shape as prices containing trading volume.
                 Used for ADV-based market impact slippage.

    Returns:
        BacktestResults with full daily state.
    """
    if config is None:
        config = BacktestConfig()

    prices = prices.sort_index()
    universe = prices.columns.tolist()
    all_dates = prices.index

    # Daily state
    n_days = len(all_dates)
    cash = np.zeros(n_days)
    positions = np.zeros((n_days, len(universe)))   # share counts
    weights_record = np.zeros((n_days, len(universe)))
    equity = np.zeros(n_days)

    cash[0] = config.initial_capital
    equity[0] = config.initial_capital

    trade_records = []
    turnover_records = []

    # Map rebalance dates to actionable execution dates (next trading day)
    rebal_set = set(rebalance_dates)

    daily_cash_factor = (1 + config.cash_return) ** (1 / 252) - 1

    for i in range(1, n_days):
        # Carry positions forward; cash accrues at risk-free rate (default 0)
        positions[i] = positions[i - 1]
        cash[i] = cash[i - 1] * (1 + daily_cash_factor)

        # Execute trades from the PREVIOUS day's signal
        prev_date = all_dates[i - 1]
        if prev_date in rebal_set:
            target_weights = signal_fn(prev_date)
            target_weights = target_weights.reindex(universe).fillna(0.0)

            # Compute current portfolio value at today's close
            today_prices = prices.iloc[i].values
            current_value = cash[i] + (positions[i] * today_prices).sum()

            # Compute target dollar exposures
            target_dollars = target_weights.values * current_value
            target_shares = np.where(today_prices > 0, target_dollars / today_prices, 0)

            # Trades: delta in shares
            trade_shares = target_shares - positions[i]
            traded_dollars = np.abs(trade_shares) * today_prices

            # Slippage: pay extra on buys, receive less on sells
            if volumes is not None:
                # ADV slippage model: slippage_cost = Base_Slippage * sqrt(trade_value / ADV_Value)
                start_idx = max(0, i - 20)
                adv_shares = volumes.iloc[start_idx:i].mean().values
                adv_shares = np.where(pd.isna(adv_shares), 1e-9, adv_shares)
                adv_dollars = adv_shares * today_prices + 1e-9
                
                # Capped square-root market impact model
                market_impact_multiplier = np.clip(np.sqrt(traded_dollars / adv_dollars), 0.0, 5.0)
                slippage_bps_dynamic = config.slippage_bps * market_impact_multiplier
                slippage_cost = traded_dollars * (slippage_bps_dynamic / 10000)
            else:
                slippage_cost = traded_dollars * (config.slippage_bps / 10000)
                
            commission_cost = traded_dollars * (config.commission_bps / 10000)
            total_cost = slippage_cost.sum() + commission_cost.sum()

            # Update positions and cash
            positions[i] = target_shares
            cash[i] = cash[i] - (trade_shares * today_prices).sum() - total_cost

            # Record trade
            tw_one_way = np.abs(trade_shares * today_prices).sum() / max(current_value, 1e-9) / 2
            turnover_records.append({'date': all_dates[i], 'turnover': tw_one_way})
            trade_records.append({
                'date': all_dates[i],
                'cost_dollars': total_cost,
                'traded_dollars': traded_dollars.sum(),
                **{f'w_{u}': target_weights.values[j] for j, u in enumerate(universe)},
            })

        # Compute equity and weights for the day
        today_prices = prices.iloc[i].values
        position_value = positions[i] * today_prices
        equity[i] = cash[i] + position_value.sum()
        if equity[i] > 0:
            weights_record[i] = position_value / equity[i]

    # Build outputs
    equity_curve = pd.Series(equity, index=all_dates, name='equity')
    cash_series = pd.Series(cash, index=all_dates, name='cash')
    positions_df = pd.DataFrame(positions, index=all_dates, columns=universe)
    weights_df = pd.DataFrame(weights_record, index=all_dates, columns=universe)

    trades_df = pd.DataFrame(trade_records).set_index('date') if trade_records else pd.DataFrame()
    turnover_series = (
        pd.DataFrame(turnover_records).set_index('date')['turnover']
        if turnover_records else pd.Series(dtype=float)
    )

    return BacktestResults(
        equity_curve=equity_curve,
        positions=positions_df,
        weights=weights_df,
        cash=cash_series,
        trades=trades_df,
        turnover=turnover_series,
    )


# ============================================================================
# REFERENCE BENCHMARK STRATEGIES
# ============================================================================

def buy_and_hold_signal(universe: list, weights: dict = None) -> Callable:
    """Constant-weight benchmark. If weights is None, equal-weight."""
    if weights is None:
        weights = {s: 1.0 / len(universe) for s in universe}
    target = pd.Series(weights)

    def signal_fn(date):
        return target.copy()
    return signal_fn


def equal_weight_signal(universe: list) -> Callable:
    """Equal-weight benchmark across the universe."""
    return buy_and_hold_signal(universe)
