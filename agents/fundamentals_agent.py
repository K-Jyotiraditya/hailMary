"""
Fundamentals Agent — Assesses financial health using yfinance data.

Fetches P/E, EPS, revenue growth, debt/equity, and free cash flow,
then prompts Gemini to produce a financial health score (0-100).
"""
import yfinance as yf
from typing import Dict, Any
from agents.base_agent import BaseAgent


class FundamentalsAgent(BaseAgent):
    NAME = "Fundamentals"

    def _fetch_fundamentals(self, ticker: str) -> dict:
        """Pull key financial metrics from yfinance."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "eps_ttm": info.get("trailingEps"),
                "revenue_growth": info.get("revenueGrowth"),
                "profit_margin": info.get("profitMargins"),
                "debt_to_equity": info.get("debtToEquity"),
                "free_cash_flow": info.get("freeCashflow"),
                "market_cap": info.get("marketCap"),
                "sector": info.get("sector", "Unknown"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "current_price": info.get("currentPrice"),
            }
        except Exception as e:
            print(f"[Fundamentals] Error for {ticker}: {e}")
            return {}

    def build_prompt(self, context: Dict[str, Any]) -> str:
        ticker = context["ticker"]
        f = context["fundamentals"]

        def fmt(v):
            if v is None: return "N/A"
            if isinstance(v, float): return f"{v:.2f}"
            if isinstance(v, int) and abs(v) > 1e6: return f"${v/1e9:.1f}B"
            return str(v)

        return f"""You are a fundamental equity analyst. Evaluate the financial health of {ticker} based on these metrics.

FINANCIAL DATA FOR {ticker}:
  Trailing P/E: {fmt(f.get('pe_ratio'))}
  Forward P/E: {fmt(f.get('forward_pe'))}
  EPS (TTM): {fmt(f.get('eps_ttm'))}
  Revenue Growth: {fmt(f.get('revenue_growth'))}
  Profit Margin: {fmt(f.get('profit_margin'))}
  Debt/Equity: {fmt(f.get('debt_to_equity'))}
  Free Cash Flow: {fmt(f.get('free_cash_flow'))}
  Market Cap: {fmt(f.get('market_cap'))}
  Sector: {f.get('sector', 'Unknown')}
  52W Range: ${fmt(f.get('52w_low'))} - ${fmt(f.get('52w_high'))}

Respond with ONLY a JSON object:
{{
  "health_score": <integer 0-100, where 100 is excellent>,
  "key_strength": "<one-line top positive factor>",
  "key_risk": "<one-line top risk factor>"
}}"""

    def parse_response(self, raw_text: str) -> Dict[str, Any]:
        data = self.extract_json(raw_text)
        # LLMs sometimes return health_score as a float or float-string (e.g. "72.5").
        # int("72.5") raises ValueError, so convert via float first.
        raw_score = data.get("health_score", 50) or 50
        try:
            health_score = int(float(raw_score))
        except (ValueError, TypeError):
            health_score = 50
        return {
            "health_score": max(0, min(100, health_score)),
            "key_strength": data.get("key_strength", ""),
            "key_risk": data.get("key_risk", ""),
        }

    def run(self, context: Dict[str, Any]):
        ticker = context["ticker"]
        fundamentals = self._fetch_fundamentals(ticker)

        if not fundamentals:
            from agents.base_agent import AgentOutput
            return AgentOutput(self.NAME, {"health_score": 50}, "No fundamental data available")

        context["fundamentals"] = fundamentals
        return super().run(context)
