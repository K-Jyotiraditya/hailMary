"""Quick 3-stock smoke test for TradingGroup V2."""
import os
import sys

import pandas as pd
from dotenv import load_dotenv

sys.path.append(".")

from agents.fundamentals_agent import FundamentalsAgent
from agents.news_sentiment import NewsSentimentAgent
from agents.portfolio_decision import PortfolioDecisionAgent
from agents.risk_style import RiskStyleAgent
from agents.technical_forecaster import TechnicalForecasterAgent
from execution.broker_live import AlpacaLiveExecution
from risk.risk_manager import PositionRisk, RiskManager

TEST_TICKERS = ["AAPL", "NVDA", "JPM"]


def main():
    load_dotenv()
    print("=== QUICK TEST: 3 stocks ===\n")
    news_agent = NewsSentimentAgent()
    tech_agent = TechnicalForecasterAgent()
    fund_agent = FundamentalsAgent()
    style_agent = RiskStyleAgent()
    portfolio_agent = PortfolioDecisionAgent()

    stock_analyses = {}
    for ticker in TEST_TICKERS:
        print(f"--- {ticker} ---")

        news = news_agent.run({"ticker": ticker})
        sentiment_score = news.data.get("sentiment_score", 0)
        sentiment_theme = news.data.get("key_theme", "?")
        print(f"  News: sent={sentiment_score:+.2f} | {sentiment_theme[:60]}")

        technical = tech_agent.run({
            "ticker": ticker,
            "sentiment_score": sentiment_score,
            "news_theme": sentiment_theme,
        })
        direction = technical.data.get("direction", "?")
        confidence = technical.data.get("confidence", 0)
        gate = technical.data.get("gate", "?")
        print(f"  Tech: {direction} (conf={confidence:.1f}) | {gate}")

        fundamentals = fund_agent.run({"ticker": ticker})
        health = fundamentals.data.get("health_score", 50)
        print(f"  Fund: health={health}/100\n")

        stock_analyses[ticker] = {
            "sentiment": news.data,
            "technical": technical.data,
            "fundamentals": fundamentals.data,
        }

    print("--- RISK STYLE ---")
    style_result = style_agent.run({
        "equity": 100000,
        "pnl_history": [],
        "current_drawdown_pct": 0,
        "recent_win_rate": 0.5,
    })
    style = style_result.data.get("style", "balanced")
    print(f"  Style: {style.upper()}\n")

    print("--- PORTFOLIO DECISION ---")
    portfolio_result = portfolio_agent.run({
        "stock_analyses": stock_analyses,
        "current_holdings": {},
        "cash_pct": 100.0,
        "trading_style": style,
        "reflection_text": "No previous history.",
    })
    weights = portfolio_result.data.get("weights", {})
    cash = portfolio_result.data.get("cash_reserve", 1.0)
    rationale = portfolio_result.data.get("rationale", "")

    print(f"\n  Weights: {weights}")
    print(f"  Cash: {cash:.1%}")
    print(f"  Rationale: {rationale}")

    print("\n--- PHASE 3: RISK MANAGEMENT ---")
    risk_mgr = RiskManager(style=style)
    positions = [
        PositionRisk(
            ticker="AAPL",
            entry_price=150.0,
            current_price=140.0,
            shares=10.0,
            entry_date="2026-05-01",
            stop_loss_pct=3.0,
            take_profit_pct=5.0,
        ),
        PositionRisk(
            ticker="NVDA",
            entry_price=100.0,
            current_price=120.0,
            shares=5.0,
            entry_date="2026-05-01",
            stop_loss_pct=5.0,
            take_profit_pct=10.0,
        ),
        PositionRisk(
            ticker="JPM",
            entry_price=100.0,
            current_price=101.0,
            shares=10.0,
            entry_date="2026-05-01",
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
        ),
    ]

    actions = risk_mgr.check_portfolio(positions, portfolio_equity=50000, starting_equity=50000)
    print(risk_mgr.summary(actions))

    print("\n--- OVERRIDDEN WEIGHTS ---")
    final_weights = risk_mgr.override_weights(weights, actions)
    print(f"  Final Weights: {final_weights}")

    print("\n--- PHASE 4: LIVE EXECUTION ---")
    alpaca_api = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY")

    if alpaca_api and alpaca_secret:
        print("  Initializing Alpaca broker connection...")
        try:
            broker = AlpacaLiveExecution(
                api_keys={"API_KEY": alpaca_api, "SECRET_KEY": alpaca_secret},
                is_paper=True,
            )
            broker.execute_target_weights(pd.Series(final_weights))
        except Exception as exc:
            print(f"  Execution failed: {exc}")
    else:
        print("  Alpaca API keys not found in .env. Skipping live execution.")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
