"""
Risk-Style Agent — Dynamically selects trading style based on recent P&L.

Reviews the last 20 trading days' performance and account state,
then selects aggressive/balanced/conservative style which controls
position sizing intensity and stop-loss/take-profit thresholds.
"""
from typing import Dict, Any
from agents.base_agent import BaseAgent


class RiskStyleAgent(BaseAgent):
    NAME = "Risk-Style"

    # Style coefficients for risk management (matching the paper)
    STYLE_PARAMS = {
        "aggressive":   {"position_scale": 1.0, "sl_mult": 2.5, "tp_mult": 3.0},
        "balanced":     {"position_scale": 0.75, "sl_mult": 2.0, "tp_mult": 2.5},
        "conservative": {"position_scale": 0.50, "sl_mult": 1.5, "tp_mult": 2.0},
    }

    def build_prompt(self, context: Dict[str, Any]) -> str:
        equity = context.get("equity", 100000)
        pnl_history = context.get("pnl_history", [])
        current_dd = context.get("current_drawdown_pct", 0)
        win_rate = context.get("recent_win_rate", 0.5)

        pnl_block = "\n".join(
            f"  Day {i+1}: {p:+.2f}%" for i, p in enumerate(pnl_history[-10:])
        ) if pnl_history else "  No trading history yet."

        return f"""You are a risk manager for a quantitative equity fund. Based on recent performance, select the most appropriate trading style for TOMORROW.

ACCOUNT STATE:
  Portfolio Equity: ${equity:,.0f}
  Current Drawdown: {current_dd:.1f}%
  Recent Win Rate (20d): {win_rate:.0%}

RECENT DAILY P&L (last 10 days):
{pnl_block}

STYLE OPTIONS:
  aggressive: Full position sizes, wider stops. Best when on a winning streak.
  balanced: 75% position sizes, moderate stops. Default safe choice.
  conservative: 50% position sizes, tight stops. Best when in drawdown.

Respond with ONLY a JSON object:
{{
  "style": "<aggressive|balanced|conservative>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one line explanation>"
}}"""

    def parse_response(self, raw_text: str) -> Dict[str, Any]:
        data = self.extract_json(raw_text)
        style = data.get("style", "balanced").lower()
        if style not in self.STYLE_PARAMS:
            style = "balanced"
        return {
            "style": style,
            "params": self.STYLE_PARAMS[style],
            "confidence": float(data.get("confidence", 0.5)),
            "reasoning": data.get("reasoning", ""),
        }
