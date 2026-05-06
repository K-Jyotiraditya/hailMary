"""
Daily Orchestrator — Runs the full TradingGroup V2 pipeline.

Sequence:
  1. For each stock: News → Technical → Fundamentals (per-stock agents)
  2. Once: Risk-Style → Portfolio-Decision (portfolio-level agents)
  3. Log all agent outputs
  4. Execute via Alpaca (if connected)
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).parent.parent))

from config.watchlist import WATCHLIST
from agents.news_sentiment import NewsSentimentAgent
from agents.technical_forecaster import TechnicalForecasterAgent
from agents.fundamentals_agent import FundamentalsAgent
from agents.risk_style import RiskStyleAgent
from agents.portfolio_decision import PortfolioDecisionAgent
from agents.base_agent import log_agent_output


def load_reflection(log_dir: str = "data/agent_logs", lookback_days: int = 5) -> str:
    """Load and summarize recent agent decisions for self-reflection."""
    log_path = Path(log_dir)
    if not log_path.exists():
        return "No previous trading history available."

    logs = sorted(log_path.glob("*.json"), reverse=True)[:lookback_days]
    if not logs:
        return "No previous trading history available."

    summary_lines = []
    for log_file in logs:
        try:
            with open(log_file) as f:
                entries = json.load(f)
            portfolio_entries = [e for e in entries if e.get("agent") == "Portfolio-Decision"]
            if portfolio_entries:
                entry = portfolio_entries[-1]
                weights = entry.get("data", {}).get("weights", {})
                rationale = entry.get("data", {}).get("rationale", "")
                summary_lines.append(
                    f"  {log_file.stem}: Allocated to {list(weights.keys())} — {rationale}"
                )
        except Exception:
            continue

    if not summary_lines:
        return "No previous portfolio decisions found."

    return "Recent decisions:\n" + "\n".join(summary_lines)


def run_daily_pipeline():
    """Execute the complete daily trading pipeline."""
    print("=" * 70)
    print(f"  TRADINGGROUP V2 — DAILY PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    news_agent = NewsSentimentAgent()
    tech_agent = TechnicalForecasterAgent()
    fund_agent = FundamentalsAgent()
    style_agent = RiskStyleAgent()
    portfolio_agent = PortfolioDecisionAgent()

    # ── Phase 1: Per-Stock Analysis ──
    print(f"\n[Phase 1] Analyzing {len(WATCHLIST)} stocks...\n")
    stock_analyses = {}

    for ticker in WATCHLIST:
        print(f"  ── {ticker} ──")

        # News
        news_out = news_agent.run({"ticker": ticker})
        log_agent_output(news_out)
        sent_data = news_out.data
        print(f"    News:  sentiment={sent_data.get('sentiment_score', 0):+.2f} | {sent_data.get('key_theme', '')[:50]}")

        # Technical (inject sentiment context)
        tech_out = tech_agent.run({
            "ticker": ticker,
            "sentiment_score": sent_data.get("sentiment_score", 0),
            "news_theme": sent_data.get("key_theme", ""),
        })
        log_agent_output(tech_out)
        tech_data = tech_out.data
        print(f"    Tech:  {tech_data.get('direction', '?')} (conf={tech_data.get('confidence', 0):.1f}) | {tech_data.get('gate', '')}")

        # Fundamentals
        fund_out = fund_agent.run({"ticker": ticker})
        log_agent_output(fund_out)
        fund_data = fund_out.data
        print(f"    Fund:  health={fund_data.get('health_score', 50)}/100")

        stock_analyses[ticker] = {
            "sentiment": sent_data,
            "technical": tech_data,
            "fundamentals": fund_data,
        }

    # ── Phase 2: Portfolio-Level Decisions ──
    print(f"\n[Phase 2] Portfolio-level intelligence...\n")

    # Risk Style
    style_out = style_agent.run({
        "equity": 100000,
        "pnl_history": [],
        "current_drawdown_pct": 0,
        "recent_win_rate": 0.5,
    })
    log_agent_output(style_out)
    style = style_out.data.get("style", "balanced")
    print(f"  Style: {style.upper()} (conf={style_out.data.get('confidence', 0):.1f})")

    # Portfolio Decision
    reflection = load_reflection()
    portfolio_out = portfolio_agent.run({
        "stock_analyses": stock_analyses,
        "current_holdings": {},
        "cash_pct": 100.0,
        "trading_style": style,
        "reflection_text": reflection,
    })
    log_agent_output(portfolio_out)

    weights = portfolio_out.data.get("weights", {})
    cash = portfolio_out.data.get("cash_reserve", 1.0)
    rationale = portfolio_out.data.get("rationale", "")

    # ── Output ──
    print(f"\n{'=' * 70}")
    print(f"  TARGET PORTFOLIO — {datetime.now().strftime('%Y-%m-%d')}")
    print(f"  Style: {style.upper()} | Cash Reserve: {cash:.1%}")
    print(f"{'=' * 70}")

    if not weights:
        print("  [ALL CASH] No positions recommended today.")
    else:
        print(f"  {'Ticker':<10} {'Weight':>8}  {'Direction':>10}  {'Sentiment':>10}  {'Health':>8}")
        print(f"  {'─'*10} {'─'*8}  {'─'*10}  {'─'*10}  {'─'*8}")
        for ticker, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            analysis = stock_analyses.get(ticker, {})
            direction = analysis.get("technical", {}).get("direction", "?")
            sentiment = analysis.get("sentiment", {}).get("sentiment_score", 0)
            health = analysis.get("fundamentals", {}).get("health_score", 50)
            print(f"  {ticker:<10} {w:>7.1%}  {direction:>10}  {sentiment:>+10.2f}  {health:>8}")

    print(f"\n  Rationale: {rationale}")
    print(f"{'=' * 70}\n")

    return weights


if __name__ == "__main__":
    weights = run_daily_pipeline()
