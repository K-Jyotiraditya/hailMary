"""
Risk Management Module — Dynamic Stop-Loss / Take-Profit + Circuit Breaker.

Implements the TradingGroup paper's risk management scheme:
  1. Per-position adaptive thresholds based on 10-day Historical Volatility
  2. Style-tiered multipliers (aggressive / balanced / conservative)
  3. Hard intercept: forced sell when PnL breaches thresholds
  4. Portfolio-level circuit breaker (max daily drawdown)

The module monitors positions and emits SELL signals when risk limits
are breached, overriding the Portfolio-Decision Agent's recommendations.
"""
import yfinance as yf
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


# Style-dependent multipliers (from the paper Section 3.6)
STYLE_MULTIPLIERS = {
    "aggressive":   {"sl_mult": 2.5, "tp_mult": 3.0},
    "balanced":     {"sl_mult": 2.0, "tp_mult": 2.5},
    "conservative": {"sl_mult": 1.5, "tp_mult": 2.0},
}

# Portfolio-level hard limits
MAX_DAILY_DRAWDOWN_PCT = 5.0    # Emergency liquidate if portfolio drops 5% intraday
MAX_SINGLE_LOSS_PCT = 8.0       # Force-sell any position down 8% from entry
MIN_POSITION_VALUE_USD = 50.0   # Don't generate orders for tiny positions


@dataclass
class PositionRisk:
    """Risk state for a single position."""
    ticker: str
    entry_price: float
    current_price: float
    shares: float
    entry_date: str
    stop_loss_pct: float    # Dynamic, based on HV
    take_profit_pct: float  # Dynamic, based on HV

    @property
    def pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100

    @property
    def market_value(self) -> float:
        return self.current_price * self.shares

    @property
    def pnl_dollars(self) -> float:
        return (self.current_price - self.entry_price) * self.shares

    @property
    def should_stop_loss(self) -> bool:
        return self.pnl_pct <= -self.stop_loss_pct

    @property
    def should_take_profit(self) -> bool:
        return self.pnl_pct >= self.take_profit_pct


@dataclass
class RiskAction:
    """An action emitted by the risk module."""
    ticker: str
    action: str          # "FORCE_SELL_SL", "FORCE_SELL_TP", "FORCE_SELL_CIRCUIT", "HOLD"
    reason: str
    pnl_pct: float
    threshold_pct: float


class RiskManager:
    """
    Dynamic risk management with per-position stop-loss/take-profit
    and portfolio-level circuit breaker.
    """

    def __init__(self, style: str = "balanced"):
        self.style = style.lower()
        if self.style not in STYLE_MULTIPLIERS:
            self.style = "balanced"
        self.mults = STYLE_MULTIPLIERS[self.style]

    def compute_hv10(self, ticker: str) -> float:
        """
        Compute 10-day unannualized daily volatility (σ_d,10).
        This matches the paper's formula exactly.
        """
        try:
            data = yf.download(ticker, period="30d", progress=False, multi_level_index=False)
            if data.empty or len(data) < 11:
                return 0.02  # Default 2% daily vol if no data
            close = data["Close"]
            log_ret = np.log(close / close.shift(1)).dropna()
            sigma_d10 = float(log_ret.tail(10).std())
            return max(sigma_d10, 0.005)  # Floor at 0.5% daily vol
        except Exception:
            return 0.02

    def compute_thresholds(self, ticker: str) -> Tuple[float, float]:
        """
        Compute adaptive stop-loss and take-profit thresholds.

        From the paper:
            T_SL = m_s^sl × σ_d,10 × 100
            T_TP = m_s^tp × σ_d,10 × 100
        """
        sigma = self.compute_hv10(ticker)
        sl_pct = self.mults["sl_mult"] * sigma * 100  # Convert to percentage
        tp_pct = self.mults["tp_mult"] * sigma * 100
        return round(sl_pct, 2), round(tp_pct, 2)

    def assess_position(self, ticker: str, entry_price: float,
                        current_price: float, shares: float,
                        entry_date: str = "") -> PositionRisk:
        """Build a PositionRisk object with dynamic thresholds."""
        sl_pct, tp_pct = self.compute_thresholds(ticker)
        return PositionRisk(
            ticker=ticker,
            entry_price=entry_price,
            current_price=current_price,
            shares=shares,
            entry_date=entry_date,
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
        )

    def check_position(self, pos: PositionRisk) -> RiskAction:
        """
        Check a single position against its dynamic thresholds.
        Returns a RiskAction.
        """
        # Hard absolute limit (regardless of style)
        if pos.pnl_pct <= -MAX_SINGLE_LOSS_PCT:
            return RiskAction(
                ticker=pos.ticker,
                action="FORCE_SELL_SL",
                reason=f"HARD LIMIT: Position down {pos.pnl_pct:.1f}% exceeds max loss {MAX_SINGLE_LOSS_PCT}%",
                pnl_pct=pos.pnl_pct,
                threshold_pct=MAX_SINGLE_LOSS_PCT,
            )

        # Dynamic stop-loss
        if pos.should_stop_loss:
            return RiskAction(
                ticker=pos.ticker,
                action="FORCE_SELL_SL",
                reason=f"STOP-LOSS: PnL {pos.pnl_pct:+.1f}% breached dynamic SL threshold -{pos.stop_loss_pct:.1f}%",
                pnl_pct=pos.pnl_pct,
                threshold_pct=pos.stop_loss_pct,
            )

        # Dynamic take-profit
        if pos.should_take_profit:
            return RiskAction(
                ticker=pos.ticker,
                action="FORCE_SELL_TP",
                reason=f"TAKE-PROFIT: PnL {pos.pnl_pct:+.1f}% exceeded TP threshold +{pos.take_profit_pct:.1f}%",
                pnl_pct=pos.pnl_pct,
                threshold_pct=pos.take_profit_pct,
            )

        return RiskAction(
            ticker=pos.ticker,
            action="HOLD",
            reason=f"Within bounds: PnL {pos.pnl_pct:+.1f}% | SL=-{pos.stop_loss_pct:.1f}% | TP=+{pos.take_profit_pct:.1f}%",
            pnl_pct=pos.pnl_pct,
            threshold_pct=0,
        )

    def check_portfolio(self, positions: List[PositionRisk],
                        portfolio_equity: float,
                        starting_equity: float) -> List[RiskAction]:
        """
        Check all positions + portfolio-level circuit breaker.
        Returns a list of RiskActions (one per position).
        """
        actions = []

        # Portfolio-level circuit breaker
        if starting_equity > 0:
            portfolio_dd = ((portfolio_equity - starting_equity) / starting_equity) * 100
            if portfolio_dd <= -MAX_DAILY_DRAWDOWN_PCT:
                # EMERGENCY: Force-sell everything
                for pos in positions:
                    actions.append(RiskAction(
                        ticker=pos.ticker,
                        action="FORCE_SELL_CIRCUIT",
                        reason=f"CIRCUIT BREAKER: Portfolio down {portfolio_dd:.1f}% exceeds {MAX_DAILY_DRAWDOWN_PCT}% daily limit. LIQUIDATING ALL.",
                        pnl_pct=pos.pnl_pct,
                        threshold_pct=MAX_DAILY_DRAWDOWN_PCT,
                    ))
                return actions

        # Per-position checks
        for pos in positions:
            if pos.market_value < MIN_POSITION_VALUE_USD:
                continue  # Skip negligible positions
            actions.append(self.check_position(pos))

        return actions

    def override_weights(self, target_weights: Dict[str, float],
                         risk_actions: List[RiskAction]) -> Dict[str, float]:
        """
        Apply risk overrides to the Portfolio-Decision Agent's target weights.
        Any position flagged for FORCE_SELL gets its weight set to 0.
        """
        overridden = dict(target_weights)
        for action in risk_actions:
            if action.action.startswith("FORCE_SELL"):
                if action.ticker in overridden:
                    overridden[action.ticker] = 0.0
        # Remove zero-weight entries
        return {k: v for k, v in overridden.items() if v > 0}

    def summary(self, actions: List[RiskAction]) -> str:
        """Human-readable summary of all risk actions."""
        lines = []
        sells = [a for a in actions if a.action.startswith("FORCE_SELL")]
        holds = [a for a in actions if a.action == "HOLD"]

        if sells:
            lines.append(f"  [!] {len(sells)} RISK OVERRIDE(S):")
            for a in sells:
                lines.append(f"    [{a.action}] {a.ticker}: {a.reason}")
        if holds:
            lines.append(f"  [OK] {len(holds)} position(s) within risk bounds")
            for a in holds:
                lines.append(f"    {a.ticker}: {a.reason}")

        return "\n".join(lines) if lines else "  No positions to monitor."


if __name__ == "__main__":
    # Smoke test
    print("=== Risk Management Module Test ===\n")
    rm = RiskManager(style="balanced")

    for ticker in ["AAPL", "NVDA", "JPM"]:
        sl, tp = rm.compute_thresholds(ticker)
        print(f"{ticker}: SL=-{sl:.2f}% | TP=+{tp:.2f}%")

    # Simulate a position in drawdown
    print("\n--- Simulated Position Check ---")
    pos = rm.assess_position("TSLA", entry_price=250.0, current_price=240.0, shares=10)
    action = rm.check_position(pos)
    print(f"TSLA: PnL={pos.pnl_pct:+.1f}% | Action={action.action} | {action.reason}")

    # Simulate take-profit
    pos2 = rm.assess_position("NVDA", entry_price=100.0, current_price=115.0, shares=5)
    action2 = rm.check_position(pos2)
    print(f"NVDA: PnL={pos2.pnl_pct:+.1f}% | Action={action2.action} | {action2.reason}")
