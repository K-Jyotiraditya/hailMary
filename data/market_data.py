"""Shared market-data helpers for Yahoo Finance access."""
from pathlib import Path
from typing import Any

import yfinance as yf
import yfinance.cache as yf_cache

_CACHE_DIR = Path(__file__).resolve().parent / "yfinance_cache"
_CACHE_INITIALIZED = False


def configure_yfinance_cache() -> Path:
    """Point yfinance's sqlite-backed caches at a repo-local writable path."""
    global _CACHE_INITIALIZED
    if not _CACHE_INITIALIZED:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        yf_cache.set_cache_location(str(_CACHE_DIR))
        _CACHE_INITIALIZED = True
    return _CACHE_DIR


def yf_download(*args: Any, **kwargs: Any):
    """Download historical market data with cache configuration applied."""
    configure_yfinance_cache()
    return yf.download(*args, **kwargs)


def yf_ticker(symbol: str) -> yf.Ticker:
    """Create a yfinance ticker object with cache configuration applied."""
    configure_yfinance_cache()
    return yf.Ticker(symbol)
