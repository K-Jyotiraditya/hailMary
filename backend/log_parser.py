"""
Log parser — reconstructs structured pipeline state from daily agent log files.

Agent entries are written sequentially: [News, Tech, Fund] × N tickers,
then [Risk-Style, Portfolio-Decision, Risk-Management] once per day.
Ticker identity is inferred from position within the per-stock block.
"""
import json
from pathlib import Path
from typing import Optional

from config.watchlist import WATCHLIST, SECTORS, MAX_SINGLE_POSITION, MAX_SECTOR_WEIGHT

PER_STOCK_AGENTS = {"News-Sentiment", "Technical-Forecaster", "Fundamentals"}
PORTFOLIO_AGENTS = {"Risk-Style", "Portfolio-Decision", "Risk-Management"}
ERROR_AGENTS = {"Pipeline-Error"}


def parse_daily_log(date_str: str) -> Optional[dict]:
    log_path = Path(f"data/agent_logs/{date_str}.json")
    if not log_path.exists():
        return None

    with open(log_path) as f:
        try:
            entries = json.load(f)
        except json.JSONDecodeError:
            return None

    per_stock = [e for e in entries if e["agent"] in PER_STOCK_AGENTS]
    portfolio = [e for e in entries if e["agent"] in PORTFOLIO_AGENTS]
    errors    = [e for e in entries if e["agent"] in ERROR_AGENTS]

    # Reconstruct per-ticker blocks: entries arrive as [N,T,F, N,T,F, ...]
    # When an agent fails silently it is never logged, so the next ticker's
    # News-Sentiment entry can appear where a Fundamentals is expected.
    # We detect that boundary and stop consuming for the current ticker.
    stock_analyses: dict = {}
    entry_idx = 0
    for ticker in WATCHLIST:
        if entry_idx >= len(per_stock):
            break
        block: dict = {}
        for slot in range(3):
            if entry_idx >= len(per_stock):
                break
            e = per_stock[entry_idx]
            agent = e["agent"]
            # A News-Sentiment at slot > 0 means a new ticker started
            if agent == "News-Sentiment" and slot > 0:
                break
            if agent == "News-Sentiment":
                block["sentiment"] = e["data"]
            elif agent == "Technical-Forecaster":
                block["technical"] = e["data"]
            elif agent == "Fundamentals":
                block["fundamentals"] = e["data"]
            entry_idx += 1
        stock_analyses[ticker] = block

    # Portfolio-level entries
    style_entry = next((e for e in portfolio if e["agent"] == "Risk-Style"), None)
    portfolio_entry = next((e for e in portfolio if e["agent"] == "Portfolio-Decision"), None)
    risk_entry = next((e for e in portfolio if e["agent"] == "Risk-Management"), None)

    weights: dict = portfolio_entry["data"].get("weights", {}) if portfolio_entry else {}

    # Derive sector weights and compliance from decided weights
    sector_weights: dict = {}
    for ticker, w in weights.items():
        sector = SECTORS.get(ticker, "Unknown")
        sector_weights[sector] = round(sector_weights.get(sector, 0.0) + w, 4)

    violations = []
    for ticker, w in weights.items():
        if w > MAX_SINGLE_POSITION:
            violations.append(f"{ticker}: {w:.1%} > {MAX_SINGLE_POSITION:.0%} single-name cap")
    for sector, w in sector_weights.items():
        if w > MAX_SECTOR_WEIGHT:
            violations.append(f"{sector}: {w:.1%} > {MAX_SECTOR_WEIGHT:.0%} sector cap")

    complete_stocks = sum(
        1 for d in stock_analyses.values()
        if all(k in d for k in ("sentiment", "technical", "fundamentals"))
    )

    last_ts = entries[-1]["timestamp"] if entries else None
    error_tickers = [e["data"].get("ticker") for e in errors if e["data"].get("ticker")]

    return {
        "date": date_str,
        "pipeline_run": {
            "phase1_complete": len(per_stock) > 0,
            "phase2_complete": portfolio_entry is not None,
            "phase3_complete": risk_entry is not None,
            "last_updated": last_ts,
            "total_entries": len(entries),
            "stocks_analyzed": len(stock_analyses),
            "stocks_complete": complete_stocks,
            "stocks_total": len(WATCHLIST),
            "error_tickers": error_tickers,
        },
        "portfolio": {
            "weights": weights,
            "cash_reserve": portfolio_entry["data"].get("cash_reserve", 1.0) if portfolio_entry else 1.0,
            "trading_style": style_entry["data"].get("style") if style_entry else None,
            "style_confidence": style_entry["data"].get("confidence") if style_entry else None,
            "rationale": portfolio_entry["data"].get("rationale") if portfolio_entry else None,
        },
        "stock_analyses": stock_analyses,
        "risk_actions": risk_entry["data"].get("actions", []) if risk_entry else [],
        "compliance": {
            "sector_weights": sector_weights,
            "single_name_max": MAX_SINGLE_POSITION,
            "sector_max": MAX_SECTOR_WEIGHT,
            "gross_exposure": round(sum(weights.values()), 4),
            "violations": violations,
        },
    }
