"""
Portfolio-Decision Agent — Synthesizes all per-stock scores into target weights.

This is the key extension over the TradingGroup paper: instead of
single-stock Buy/Hold/Sell, this agent outputs a multi-stock portfolio
weight allocation respecting sector caps and position limits.
"""
import json
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from config.watchlist import MAX_SINGLE_POSITION, MAX_SECTOR_WEIGHT, MAX_STOCKS_HELD, SECTORS


class PortfolioDecisionAgent(BaseAgent):
    NAME = "Portfolio-Decision"

    def build_prompt(self, context: Dict[str, Any]) -> str:
        stock_analyses = context["stock_analyses"]
        current_holdings = context.get("current_holdings", {})
        cash_pct = context.get("cash_pct", 100.0)
        reflection = context.get("reflection_text", "No previous trading history.")
        style = context.get("trading_style", "balanced")

        analysis_block = ""
        for ticker, analysis in stock_analyses.items():
            sent = analysis.get("sentiment", {})
            tech = analysis.get("technical", {})
            fund = analysis.get("fundamentals", {})
            sector = SECTORS.get(ticker, "Unknown")
            analysis_block += f"""
  {ticker} ({sector}):
    Sentiment: {sent.get('sentiment_score', 0):.2f} (conf: {sent.get('confidence', 0):.1f}) — {sent.get('key_theme', 'N/A')}
    Technical: {tech.get('direction', 'N/A')} (conf: {tech.get('confidence', 0):.1f}) | Gate: {tech.get('gate', 'N/A')}
    Fundamentals: Health {fund.get('health_score', 50)}/100 | Strength: {fund.get('key_strength', 'N/A')}
"""

        holdings_block = json.dumps(current_holdings, indent=2) if current_holdings else "None (100% cash)"

        return f"""You are a portfolio manager for a US large-cap equity fund. Based on the analysis from your research team, construct the optimal portfolio allocation for TOMORROW.

CONSTRAINTS (MANDATORY):
- Max {MAX_STOCKS_HELD} stocks in portfolio
- Max {MAX_SINGLE_POSITION*100:.0f}% per single stock
- Max {MAX_SECTOR_WEIGHT*100:.0f}% per sector
- Weights must sum to <= 1.0 (remainder is cash)
- Trading style: {style} (aggressive=higher conviction, conservative=more cash)

CURRENT HOLDINGS:
{holdings_block}

CASH AVAILABLE: {cash_pct:.1f}%

RESEARCH TEAM ANALYSIS:
{analysis_block}

PAST TRADING EXPERIENCE:
{reflection}

Output ONLY a JSON object with target weights (0 for stocks to skip, omit stocks not in the list):
{{
  "weights": {{"TICKER": <weight as float>, ...}},
  "cash_reserve": <float, remainder that stays in cash>,
  "rationale": "<brief explanation of allocation logic>"
}}"""

    def parse_response(self, raw_text: str) -> Dict[str, Any]:
        data = self.extract_json(raw_text)
        weights = data.get("weights", {})

        # Enforce constraints
        cleaned = {}
        for ticker, w in weights.items():
            w = max(0.0, min(float(w), MAX_SINGLE_POSITION))
            if w > 0.005:  # Ignore negligible weights
                cleaned[ticker] = round(w, 4)

        # Enforce sector caps
        sector_totals = {}
        for ticker, w in cleaned.items():
            sec = SECTORS.get(ticker, "Other")
            sector_totals[sec] = sector_totals.get(sec, 0) + w

        for sec, total in sector_totals.items():
            if total > MAX_SECTOR_WEIGHT:
                scale = MAX_SECTOR_WEIGHT / total
                for ticker in list(cleaned.keys()):
                    if SECTORS.get(ticker, "Other") == sec:
                        cleaned[ticker] = round(cleaned[ticker] * scale, 4)

        # Cap total at 1.0
        total = sum(cleaned.values())
        if total > 1.0:
            scale = 1.0 / total
            cleaned = {k: round(v * scale, 4) for k, v in cleaned.items()}

        # Limit to MAX_STOCKS_HELD
        if len(cleaned) > MAX_STOCKS_HELD:
            sorted_w = sorted(cleaned.items(), key=lambda x: x[1], reverse=True)
            cleaned = dict(sorted_w[:MAX_STOCKS_HELD])

        return {
            "weights": cleaned,
            "cash_reserve": round(1.0 - sum(cleaned.values()), 4),
            "rationale": data.get("rationale", ""),
        }
