"""
Daily Orchestrator — Runs the full TradingGroup V2 pipeline.

Sequence:
  1. For each stock: News → Technical → Fundamentals (per-stock agents)
  2. Once: Risk-Style → Portfolio-Decision (portfolio-level agents)
  3. Risk Management: check existing positions, override weights if needed
  4. Log all agent outputs
  5. Execute via Alpaca (if connected)
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
from agents.base_agent import log_agent_output, AgentOutput
from risk.risk_manager import RiskManager


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


def load_current_positions() -> dict:
    """Load simulated positions from the latest execution log."""
    state_path = Path("data/portfolio_state.json")
    if state_path.exists():
        try:
            with open(state_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"positions": {}, "equity": 100000, "starting_equity": 100000}


def save_portfolio_state(weights: dict, equity: float = 100000):
    """Save the portfolio state for next-day risk monitoring."""
    import yfinance as yf

    positions = {}
    for ticker, w in weights.items():
        if w <= 0:
            continue
        try:
            data = yf.download(ticker, period="1d", progress=False, multi_level_index=False)
            if not data.empty:
                price = float(data["Close"].iloc[-1])
                dollar_alloc = equity * w
                shares = dollar_alloc / price
                positions[ticker] = {
                    "entry_price": price,
                    "shares": round(shares, 2),
                    "entry_date": datetime.now().strftime("%Y-%m-%d"),
                }
        except Exception:
            continue

    state = {
        "positions": positions,
        "equity": equity,
        "starting_equity": equity,
        "last_update": datetime.now().isoformat(),
    }
    Path("data").mkdir(exist_ok=True)
    with open("data/portfolio_state.json", "w") as f:
        json.dump(state, f, indent=2)


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

    # ── Phase 3: Risk Management ──
    print(f"\n[Phase 3] Risk management checks...\n")
    risk_mgr = RiskManager(style=style)
    state = load_current_positions()

    if state["positions"]:
        import yfinance as yf
        risk_positions = []
        for ticker, pos_data in state["positions"].items():
            try:
                data = yf.download(ticker, period="1d", progress=False, multi_level_index=False)
                current_price = float(data["Close"].iloc[-1]) if not data.empty else pos_data["entry_price"]
            except Exception:
                current_price = pos_data["entry_price"]

            risk_pos = risk_mgr.assess_position(
                ticker=ticker,
                entry_price=pos_data["entry_price"],
                current_price=current_price,
                shares=pos_data["shares"],
                entry_date=pos_data.get("entry_date", ""),
            )
            risk_positions.append(risk_pos)

        actions = risk_mgr.check_portfolio(
            risk_positions,
            portfolio_equity=state["equity"],
            starting_equity=state["starting_equity"],
        )
        print(risk_mgr.summary(actions))

        # Override weights if risk module flags positions
        original_count = len(weights)
        weights = risk_mgr.override_weights(weights, actions)
        overrides = original_count - len(weights)
        if overrides > 0:
            print(f"\n  ⚠️  Risk module removed {overrides} position(s) from target weights")

        # Log risk actions
        risk_log = AgentOutput(
            agent_name="Risk-Management",
            data={"actions": [{"ticker": a.ticker, "action": a.action, "pnl": a.pnl_pct} for a in actions]},
            reasoning=risk_mgr.summary(actions),
        )
        log_agent_output(risk_log)
    else:
        print("  No existing positions to monitor. First run — all clear.")

        # Show what thresholds WOULD be for the proposed portfolio
        if weights:
            print(f"\n  Proposed position thresholds ({style.upper()} mode):")
            for ticker in list(weights.keys())[:5]:
                sl, tp = risk_mgr.compute_thresholds(ticker)
                print(f"    {ticker}: SL=-{sl:.1f}% | TP=+{tp:.1f}%")

    # ── Output ──
    print(f"\n{'=' * 70}")
    print(f"  FINAL TARGET PORTFOLIO — {datetime.now().strftime('%Y-%m-%d')}")
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

    # Save state for tomorrow's risk monitoring
    if weights:
        save_portfolio_state(weights)
        print("  [STATE] Portfolio state saved for next-day risk monitoring.\n")

    # ── Phase 4: Live Execution ──
    print(f"\n[Phase 4] Alpaca Paper Execution...\n")
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
            target_series = pd.Series(weights)
            broker.execute_target_weights(target_series)
        except Exception as e:
            print(f"  ❌ Alpaca execution failed: {e}")
    else:
        print("  ⚠️ Alpaca API keys not found in .env. Skipping live execution.")


    return weights


if __name__ == "__main__":
    weights = run_daily_pipeline()
