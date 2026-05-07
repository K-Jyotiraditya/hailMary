"""
Daily Orchestrator — Runs the full TradingGroup V2 pipeline.

Sequence:
  0. Macro regime gate + universe screen (weekly)
  1. For each stock: News -> Technical -> Fundamentals (per-stock agents)
  2. Once: Risk-Style -> HRP weights -> Portfolio-Decision overlay
  3. Risk Management: check existing positions, override weights if needed
  4. Log all agent outputs
  5. Execute via Alpaca (if connected)
"""
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from data.market_data import yf_download
from data.vector_memory import VectorMemory
from risk.risk_manager import RiskManager


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

    # Preserve the original starting_equity across saves so the circuit breaker
    # can detect cumulative drawdown from the account baseline, not just today.
    existing = load_current_positions()
    starting_equity = existing.get("starting_equity", equity)

    positions = {}
    for ticker, w in weights.items():
        if w <= 0:
            continue
        try:
            data = yf_download(ticker, period="1d", progress=False, multi_level_index=False)
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
        "starting_equity": starting_equity,
        "last_update": datetime.now().isoformat(),
    }
    Path("data").mkdir(exist_ok=True)
    with open("data/portfolio_state.json", "w") as f:
        json.dump(state, f, indent=2)


def check_ollama() -> bool:
    """Return True if the local Ollama service is reachable."""
    import requests
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _analyze_stock(ticker: str, session_date: str) -> tuple:
    """Worker: analyze one stock with its own agent instances (thread-safe)."""
    news_agent = NewsSentimentAgent()
    tech_agent = TechnicalForecasterAgent()
    fund_agent = FundamentalsAgent()
    try:
        news_out = news_agent.run({"ticker": ticker})
        news_out.data["ticker"] = ticker
        log_agent_output(news_out, date_str=session_date)
        sent_data = news_out.data

        tech_out = tech_agent.run({
            "ticker": ticker,
            "sentiment_score": sent_data.get("sentiment_score", 0),
            "news_theme": sent_data.get("key_theme", ""),
        })
        tech_out.data["ticker"] = ticker
        log_agent_output(tech_out, date_str=session_date)
        tech_data = tech_out.data

        fund_out = fund_agent.run({"ticker": ticker})
        fund_out.data["ticker"] = ticker
        log_agent_output(fund_out, date_str=session_date)
        fund_data = fund_out.data

        result = {
            "sentiment": sent_data,
            "technical": tech_data,
            "fundamentals": fund_data,
        }
        summary = (
            f"  -- {ticker} -- "
            f"News:{sent_data.get('sentiment_score', 0):+.2f} | "
            f"Tech:{tech_data.get('direction', '?')}(conf={tech_data.get('confidence', 0):.1f}) | "
            f"Fund:{fund_data.get('health_score', 50)}/100"
        )
        return ticker, result, summary, None

    except Exception as exc:
        err_out = AgentOutput(
            agent_name="Pipeline-Error",
            data={"ticker": ticker, "error": str(exc)},
            reasoning=f"Unhandled exception during {ticker} analysis",
        )
        log_agent_output(err_out, date_str=session_date)
        fallback = {
            "sentiment":    {"sentiment_score": 0.0, "confidence": 0.0, "key_theme": "analysis failed"},
            "technical":    {"direction": "sideways", "confidence": 0.0, "gate": "ERROR"},
            "fundamentals": {"health_score": 50, "key_strength": "", "key_risk": ""},
        }
        return ticker, fallback, f"  -- {ticker} -- [ERROR] {exc}", exc


def run_daily_pipeline():
    """Execute the complete daily trading pipeline."""
    session_date = datetime.now().strftime('%Y-%m-%d')
    print("=" * 70)
    print(f"  TRADINGGROUP V2 - DAILY PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if not check_ollama():
        print("\n  [FATAL] Ollama is not running on localhost:11434.")
        print("  Start it with: ollama serve")
        print("  Then retry:    python orchestrator/daily_pipeline.py\n")
        return {}

    # ── Phase 0: Macro Regime + Universe Screen ──
    print(f"\n[Phase 0] Macro regime detection + universe screening...\n")

    macro = {"regime": "neutral", "equity_mult": 0.7, "signals": []}
    try:
        from data.macro_regime import get_macro_regime
        macro = get_macro_regime()
        print(f"  Macro regime: {macro['regime'].upper()} "
              f"(equity_mult={macro['equity_mult']:.1f}) "
              f"| VIX={macro['vix']:.1f} "
              f"| YieldSpread={macro['yield_spread']:.3f} "
              f"| Credit={macro['credit_spread']:.3f}")
        print(f"  Signals: {', '.join(macro['signals'])}")
    except Exception as e:
        print(f"  [Macro] Failed ({e}) - defaulting to neutral")

    # Universe screen: use cached weekly candidates, fallback to static watchlist
    active_universe = WATCHLIST
    try:
        from config.universe import get_universe
        active_universe = get_universe()
        print(f"\n  Universe: {len(active_universe)} stocks selected from S&P 500 screen")
    except Exception as e:
        print(f"  [Universe] Failed ({e}) - using static watchlist ({len(WATCHLIST)} stocks)")

    if macro["regime"] == "risk_off":
        print(f"\n  [!] RISK-OFF regime detected — reducing equity exposure to "
              f"{macro['equity_mult']:.0%}. Watchlist trimmed to top 10 by momentum.")
        # In risk-off: only run top 10 most liquid/safest stocks
        active_universe = active_universe[:10]

    style_agent = RiskStyleAgent()
    portfolio_agent = PortfolioDecisionAgent()

    # ── Phase 1: Per-Stock Analysis (parallel) ──
    print(f"\n[Phase 1] Analyzing {len(active_universe)} stocks in parallel (5 workers)...\n")
    stock_analyses = {}

    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        for ticker in active_universe:
            futures_map[pool.submit(_analyze_stock, ticker, session_date)] = ticker

        for future in as_completed(futures_map):
            ticker_result, analysis, summary, error = future.result()
            print(summary)
            stock_analyses[ticker_result] = analysis

    # Load portfolio state once — shared by Phase 2 and Phase 3
    state = load_current_positions()
    equity = state.get("equity", 100000)

    # Derive current holdings as weight fractions from entry prices (best available estimate)
    current_holdings_pct: dict = {}
    cash_pct = 100.0
    if state["positions"]:
        total_invested = sum(
            p["entry_price"] * p["shares"] for p in state["positions"].values()
        )
        cash_pct = max(0.0, round((equity - total_invested) / equity * 100, 1))
        current_holdings_pct = {
            ticker: round(p["entry_price"] * p["shares"] / equity, 4)
            for ticker, p in state["positions"].items()
        }

    # ── Phase 2: Portfolio-Level Decisions ──
    print(f"\n[Phase 2] Portfolio-level intelligence...\n")

    weights = {}
    cash = 1.0
    rationale = ""
    style = "balanced"

    try:
        # Risk Style
        style_out = style_agent.run({
            "equity": equity,
            "pnl_history": [],
            "current_drawdown_pct": 0,
            "recent_win_rate": 0.5,
        })
        log_agent_output(style_out, date_str=session_date)
        style = style_out.data.get("style", "balanced")
        print(f"  Style: {style.upper()} (conf={style_out.data.get('confidence', 0):.1f})")

        # Options flow signals (live, parallel fetch)
        print("  Fetching options flow signals...")
        options_signals = {}
        try:
            from data.options_flow import get_options_signal
            from concurrent.futures import ThreadPoolExecutor as _TPE
            with _TPE(max_workers=5) as _pool:
                _futs = {_pool.submit(get_options_signal, t): t for t in stock_analyses}
                for _f in _futs:
                    options_signals[_futs[_f]] = _f.result()
            bullish_options = [t for t, s in options_signals.items()
                               if s.get("options_composite", 0) > 0.2]
            print(f"  Options: {len(bullish_options)} bullish flow signals")
        except Exception as e:
            print(f"  [OPTIONS] Failed (non-fatal): {e}")

        # Pairs trading signals
        pairs_adjustments = {}
        try:
            from data.pairs_trading import find_cointegrated_pairs, get_pairs_signals, pairs_to_weight_adjustments
            pairs = find_cointegrated_pairs(list(stock_analyses.keys()))
            if pairs:
                pair_sigs = get_pairs_signals(pairs)
                pairs_adjustments = pairs_to_weight_adjustments(pair_sigs)
                if pairs_adjustments:
                    print(f"  Pairs: long signals for {list(pairs_adjustments.keys())}")
        except Exception as e:
            print(f"  [PAIRS] Failed (non-fatal): {e}")

        # HRP base weights — mathematically diversified allocation
        hrp_weights = {}
        try:
            from models.hrp import compute_hrp_weights
            candidate_tickers = list(stock_analyses.keys())
            print(f"  Computing HRP weights for {len(candidate_tickers)} candidates...")
            hrp_weights = compute_hrp_weights(candidate_tickers)
            print(f"  HRP complete. Top 5: " +
                  ", ".join(f"{t}={w:.1%}" for t, w in
                             sorted(hrp_weights.items(), key=lambda x: x[1], reverse=True)[:5]))
        except Exception as e:
            print(f"  [HRP] Failed ({e}) - LLM will determine weights")

        # Portfolio Decision — use vector RAG for reflection context
        vm = VectorMemory()
        reflection_query = f"portfolio allocation {style} style {session_date}"
        reflection = vm.retrieve_similar(reflection_query, n_results=5)
        portfolio_out = portfolio_agent.run({
            "stock_analyses":   stock_analyses,
            "current_holdings": current_holdings_pct,
            "cash_pct":         cash_pct,
            "trading_style":    style,
            "reflection_text":  reflection,
            "options_signals":  options_signals,
            "hrp_weights":      hrp_weights,
        })
        log_agent_output(portfolio_out, date_str=session_date)

        weights = portfolio_out.data.get("weights", {})

        # If LLM returned empty weights, fall back to HRP
        if not weights and hrp_weights:
            print("  [!] LLM returned no weights — using HRP weights directly")
            weights = hrp_weights

        cash = portfolio_out.data.get("cash_reserve", 1.0)
        rationale = portfolio_out.data.get("rationale", "")

        # Apply pairs trading overlays (cap total weight at 1.0)
        if pairs_adjustments:
            for t, delta in pairs_adjustments.items():
                weights[t] = weights.get(t, 0.0) + delta
            total_w = sum(weights.values())
            if total_w > 1.0:
                weights = {t: w / total_w for t, w in weights.items()}

        # Apply macro regime equity multiplier (scale down in neutral/risk-off)
        equity_mult = macro.get("equity_mult", 1.0)
        if equity_mult < 1.0 and weights:
            weights = {t: w * equity_mult for t, w in weights.items()}
            regime = macro.get("regime", "neutral")
            print(f"\n  [Macro] {regime.upper()} regime: equity weights scaled to "
                  f"{equity_mult:.0%} (remaining {1 - equity_mult:.0%} held as cash)")

    except Exception as exc:
        print(f"  [ERROR] Phase 2 failed: {exc} - defaulting to all-cash")
        err_out = AgentOutput(
            agent_name="Pipeline-Error",
            data={"phase": 2, "error": str(exc)},
            reasoning="Unhandled exception in Phase 2",
        )
        log_agent_output(err_out, date_str=session_date)

    # ── Phase 3: Risk Management ──
    print(f"\n[Phase 3] Risk management checks...\n")
    risk_mgr = RiskManager(style=style)

    if state["positions"]:
        risk_positions = []
        for ticker, pos_data in state["positions"].items():
            try:
                data = yf_download(ticker, period="1d", progress=False, multi_level_index=False)
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
            print(f"\n  [!] Risk module removed {overrides} position(s) from target weights")

        # Log risk actions
        risk_log = AgentOutput(
            agent_name="Risk-Management",
            data={"actions": [{"ticker": a.ticker, "action": a.action, "pnl": a.pnl_pct} for a in actions]},
            reasoning=risk_mgr.summary(actions),
        )
        log_agent_output(risk_log, date_str=session_date)
    else:
        print("  No existing positions to monitor. First run - all clear.")

        # Show what thresholds WOULD be for the proposed portfolio
        if weights:
            print(f"\n  Proposed position thresholds ({style.upper()} mode):")
            for ticker in list(weights.keys())[:5]:
                sl, tp = risk_mgr.compute_thresholds(ticker)
                print(f"    {ticker}: SL=-{sl:.1f}% | TP=+{tp:.1f}%")

        # Always log a Risk-Management entry so Phase 3 shows complete in the dashboard
        risk_log = AgentOutput(
            agent_name="Risk-Management",
            data={"actions": []},
            reasoning="No existing positions. Risk checks not required.",
        )
        log_agent_output(risk_log, date_str=session_date)

    # ── Output ──
    print(f"\n{'=' * 70}")
    print(f"  FINAL TARGET PORTFOLIO - {datetime.now().strftime('%Y-%m-%d')}")
    print(f"  Style: {style.upper()} | Cash Reserve: {cash:.1%}")
    print(f"{'=' * 70}")

    if not weights:
        print("  [ALL CASH] No positions recommended today.")
    else:
        print(f"  {'Ticker':<10} {'Weight':>8}  {'Direction':>10}  {'Sentiment':>10}  {'Health':>8}")
        print(f"  {'-'*10} {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")
        for ticker, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            analysis = stock_analyses.get(ticker, {})
            direction = analysis.get("technical", {}).get("direction", "?")
            sentiment = analysis.get("sentiment", {}).get("sentiment_score", 0)
            health = analysis.get("fundamentals", {}).get("health_score", 50)
            print(f"  {ticker:<10} {w:>7.1%}  {direction:>10}  {sentiment:>+10.2f}  {health:>8}")

    print(f"\n  Rationale: {rationale}")
    print(f"{'=' * 70}\n")

    # Save state for tomorrow's risk monitoring + vector memory
    if weights:
        save_portfolio_state(weights, equity)
        print("  [STATE] Portfolio state saved for next-day risk monitoring.\n")
        try:
            vm = VectorMemory()
            vm.store_decision(
                date=session_date,
                weights=weights,
                rationale=rationale,
                style=style,
            )
            print(f"  [MEMORY] Decision stored in vector memory ({vm.count()} total).\n")
        except Exception as e:
            print(f"  [MEMORY] Vector store failed (non-fatal): {e}\n")

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
            print(f"  [ERROR] Alpaca execution failed: {e}")
    else:
        print("  [!] Alpaca API keys not found in .env. Skipping live execution.")


    return weights


if __name__ == "__main__":
    weights = run_daily_pipeline()
