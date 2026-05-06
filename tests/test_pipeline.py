"""Quick 3-stock smoke test for TradingGroup V2."""
import sys
sys.path.append(".")
from dotenv import load_dotenv
load_dotenv()

from agents.news_sentiment import NewsSentimentAgent
from agents.technical_forecaster import TechnicalForecasterAgent
from agents.fundamentals_agent import FundamentalsAgent
from agents.risk_style import RiskStyleAgent
from agents.portfolio_decision import PortfolioDecisionAgent

TEST_TICKERS = ["AAPL", "NVDA", "JPM"]

print("=== QUICK TEST: 3 stocks ===\n")
news_agent = NewsSentimentAgent()
tech_agent = TechnicalForecasterAgent()
fund_agent = FundamentalsAgent()
style_agent = RiskStyleAgent()
portfolio_agent = PortfolioDecisionAgent()

stock_analyses = {}
for t in TEST_TICKERS:
    print(f"--- {t} ---")

    n = news_agent.run({"ticker": t})
    s_score = n.data.get("sentiment_score", 0)
    s_theme = n.data.get("key_theme", "?")
    print(f"  News: sent={s_score:+.2f} | {s_theme[:60]}")

    tc = tech_agent.run({
        "ticker": t,
        "sentiment_score": s_score,
        "news_theme": s_theme,
    })
    direction = tc.data.get("direction", "?")
    conf = tc.data.get("confidence", 0)
    gate = tc.data.get("gate", "?")
    print(f"  Tech: {direction} (conf={conf:.1f}) | {gate}")

    f = fund_agent.run({"ticker": t})
    health = f.data.get("health_score", 50)
    print(f"  Fund: health={health}/100\n")

    stock_analyses[t] = {
        "sentiment": n.data,
        "technical": tc.data,
        "fundamentals": f.data,
    }

print("--- RISK STYLE ---")
s = style_agent.run({
    "equity": 100000,
    "pnl_history": [],
    "current_drawdown_pct": 0,
    "recent_win_rate": 0.5,
})
style = s.data.get("style", "balanced")
print(f"  Style: {style.upper()}\n")

print("--- PORTFOLIO DECISION ---")
p = portfolio_agent.run({
    "stock_analyses": stock_analyses,
    "current_holdings": {},
    "cash_pct": 100.0,
    "trading_style": style,
    "reflection_text": "No previous history.",
})
weights = p.data.get("weights", {})
cash = p.data.get("cash_reserve", 1.0)
rationale = p.data.get("rationale", "")

print(f"  Weights: {weights}")
print(f"  Cash: {cash:.1%}")
print(f"  Rationale: {rationale}")
print("\n=== DONE ===")
