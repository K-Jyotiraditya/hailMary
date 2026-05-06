"""
Watchlist Configuration — defines the stock universe for TradingGroup V2.

Top 15 S&P 100 constituents by average daily volume and analyst coverage.
These are the most liquid, most news-covered US large caps — ideal for
LLM-driven sentiment analysis where news availability is critical.
"""

WATCHLIST = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "NVDA",   # NVIDIA
    "AMZN",   # Amazon
    "GOOGL",  # Alphabet
    "META",   # Meta
    "TSLA",   # Tesla
    "JPM",    # JPMorgan
    "V",      # Visa
    "UNH",    # UnitedHealth
    "JNJ",    # Johnson & Johnson
    "XOM",    # Exxon Mobil
    "PG",     # Procter & Gamble
    "HD",     # Home Depot
    "NFLX",   # Netflix
]

BENCHMARK = "^GSPC"  # S&P 500

# Sector mapping for portfolio constraints
SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "META": "Technology",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "NFLX": "Communication Services",
    "JPM": "Financials", "V": "Financials",
    "UNH": "Healthcare", "JNJ": "Healthcare",
    "XOM": "Energy",
    "PG": "Consumer Staples",
}

# Portfolio constraints
MAX_SINGLE_POSITION = 0.15   # 15% max per stock
MAX_SECTOR_WEIGHT = 0.35     # 35% max per sector
MAX_STOCKS_HELD = 10         # Hold at most 10 at once
