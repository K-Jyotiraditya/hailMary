"""
HailMary Backend — FastAPI server exposing pipeline state to the dashboard.

Run from project root:
    uvicorn backend.server:app --reload --port 8000
"""
import json
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.log_parser import parse_daily_log
from config.watchlist import WATCHLIST, SECTORS

app = FastAPI(title="HailMary API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:4173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

LOG_DIR = Path("data/agent_logs")
PORTFOLIO_STATE = Path("data/portfolio_state.json")


# ── Health ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ── Pipeline status ──────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    """Return today's parsed pipeline state. Falls back to a zero-state if no log exists."""
    today = date.today().strftime("%Y-%m-%d")
    result = parse_daily_log(today)
    if result is None:
        return _empty_status(today)
    return result


@app.get("/api/logs/{date_str}")
def get_log(date_str: str):
    """Return parsed pipeline state for a specific date (YYYY-MM-DD)."""
    result = parse_daily_log(date_str)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No log found for {date_str}")
    return result


@app.get("/api/dates")
def get_dates():
    """Return all available log dates, newest first."""
    if not LOG_DIR.exists():
        return []
    return sorted([f.stem for f in LOG_DIR.glob("*.json")], reverse=True)


# ── Portfolio state ──────────────────────────────────────────────────────────

@app.get("/api/portfolio")
def get_portfolio():
    """Return the saved portfolio state used for next-day risk monitoring."""
    if not PORTFOLIO_STATE.exists():
        return {"positions": {}, "equity": 100000, "starting_equity": 100000, "last_update": None}
    with open(PORTFOLIO_STATE) as f:
        return json.load(f)


# ── Reference data ───────────────────────────────────────────────────────────

@app.get("/api/watchlist")
def get_watchlist():
    """Return the current watchlist with sector mappings."""
    return {
        "tickers": WATCHLIST,
        "sectors": SECTORS,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _empty_status(today: str) -> dict:
    return {
        "date": today,
        "pipeline_run": {
            "phase1_complete": False,
            "phase2_complete": False,
            "phase3_complete": False,
            "last_updated": None,
            "total_entries": 0,
            "stocks_analyzed": 0,
            "stocks_complete": 0,
            "stocks_total": len(WATCHLIST),
        },
        "portfolio": {
            "weights": {},
            "cash_reserve": 1.0,
            "trading_style": None,
            "style_confidence": None,
            "rationale": None,
        },
        "stock_analyses": {},
        "risk_actions": [],
        "compliance": {
            "sector_weights": {},
            "single_name_max": 0.15,
            "sector_max": 0.35,
            "gross_exposure": 0.0,
            "violations": [],
        },
    }
