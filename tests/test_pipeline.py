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
from risk.risk_manager import RiskManager

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

print(f"\n  Weights: {weights}")
print(f"  Cash: {cash:.1%}")
print(f"  Rationale: {rationale}")

print("\n--- PHASE 3: RISK MANAGEMENT ---")
risk_mgr = RiskManager(style=style)
# Simulate dummy positions for the risk test
from risk.risk_manager import PositionRisk

positions = [
    PositionRisk(ticker="AAPL", entry_price=150.0, current_price=140.0, shares=10.0, entry_date="2026-05-01", stop_loss_pct=3.0, take_profit_pct=5.0), # Will hit stop loss
    PositionRisk(ticker="NVDA", entry_price=100.0, current_price=120.0, shares=5.0, entry_date="2026-05-01", stop_loss_pct=5.0, take_profit_pct=10.0), # Will hit take profit
    PositionRisk(ticker="JPM", entry_price=100.0, current_price=101.0, shares=10.0, entry_date="2026-05-01", stop_loss_pct=2.0, take_profit_pct=4.0), # Hold
]

actions = risk_mgr.check_portfolio(positions, portfolio_equity=50000, starting_equity=50000)
print(risk_mgr.summary(actions))

print("\n--- OVERRIDDEN WEIGHTS ---")
final_weights = risk_mgr.override_weights(weights, actions)
print(f"  Final Weights: {final_weights}")

print("\n--- PHASE 4: LIVE EXECUTION ---")
import os
import pandas as pd
from execution.broker_live import AlpacaLiveExecution

alpaca_api = os.getenv("ALPACA_API_KEY")
alpaca_secret = os.getenv("ALPACA_SECRET_KEY")

if alpaca_api and alpaca_secret:
    print("  Initializing Alpaca broker connection...")
    try:
        broker = AlpacaLiveExecution(
            api_keys={"API_KEY": alpaca_api, "SECRET_KEY": alpaca_secret},
            is_paper=True
        )
        target_series = pd.Series(final_weights)
        broker.execute_target_weights(target_series)
    except Exception as e:
        print(f"  ❌ Alpaca execution failed: {e}")
else:
    print("  ⚠️ Alpaca API keys not found in .env. Skipping live execution.")

print("\n=== DONE ===")
