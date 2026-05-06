"""
NIFTY 46 Sector Classification — manually mapped from official NSE sector tags.

Used by the sector-neutral portfolio constraint to prevent concentration in
any single sector (e.g., today's top-10 was 70% IT + Banking).

Sectors are kept coarse — detailed GICS-style sub-industries don't help much
when you only have 46 stocks. Eight buckets is enough granularity.
"""

# Sector classification: maps stock symbol -> sector name
SECTORS = {
    # Banking & Finance
    'HDFCBANK':   'Financials',
    'ICICIBANK':  'Financials',
    'AXISBANK':   'Financials',
    'KOTAKBANK':  'Financials',
    'INDUSINDBK': 'Financials',
    'SBIN':       'Financials',
    'BAJFINANCE': 'Financials',
    'BAJAJFINSV': 'Financials',
    'SBILIFE':    'Financials',
    'HDFCLIFE':   'Financials',
    'SHRIRAMFIN': 'Financials',

    # Information Technology
    'TCS':        'IT',
    'INFY':       'IT',
    'HCLTECH':    'IT',
    'WIPRO':      'IT',
    'TECHM':      'IT',

    # Energy & Utilities
    'RELIANCE':   'Energy',
    'ONGC':       'Energy',
    'BPCL':       'Energy',
    'COALINDIA':  'Energy',
    'NTPC':       'Utilities',
    'POWERGRID':  'Utilities',
    'ADANIPORTS': 'Utilities',  # Infrastructure / ports
    'ADANIENT':   'Utilities',  # Diversified infra

    # Consumer
    'HINDUNILVR': 'Consumer',
    'ITC':        'Consumer',
    'NESTLEIND':  'Consumer',
    'BRITANNIA':  'Consumer',
    'TATACONSUM': 'Consumer',
    'TITAN':      'Consumer',
    'ASIANPAINT': 'Consumer',

    # Auto
    'MARUTI':     'Auto',
    'M&M':        'Auto',
    'EICHERMOT':  'Auto',
    'BAJAJ-AUTO': 'Auto',

    # Healthcare
    'SUNPHARMA':  'Healthcare',
    'CIPLA':      'Healthcare',
    'DRREDDY':    'Healthcare',
    'DIVISLAB':   'Healthcare',
    'APOLLOHOSP': 'Healthcare',

    # Industrials & Materials
    'LT':         'Industrials',
    'ULTRACEMCO': 'Materials',
    'GRASIM':     'Materials',
    'TATASTEEL':  'Materials',
    'JSWSTEEL':   'Materials',

    # Telecom
    'BHARTIARTL': 'Telecom',
}


def get_sector(symbol: str) -> str:
    """Return sector for a given symbol; 'Other' if not in our mapping."""
    return SECTORS.get(symbol, 'Other')


def get_sector_map(symbols=None) -> dict:
    """Return symbol -> sector dict, optionally restricted to a set of symbols."""
    if symbols is None:
        return dict(SECTORS)
    return {s: get_sector(s) for s in symbols}


def all_sectors() -> list:
    """List of all unique sectors."""
    return sorted(set(SECTORS.values()))


def stocks_by_sector(sector: str) -> list:
    """All stocks belonging to a given sector."""
    return [s for s, sec in SECTORS.items() if sec == sector]


def sector_distribution(weights) -> dict:
    """
    Given a Series/dict of weights, return total weight per sector.
    """
    import pandas as pd
    if isinstance(weights, dict):
        weights = pd.Series(weights)
    by_sector = {}
    for symbol, w in weights.items():
        sec = get_sector(symbol)
        by_sector[sec] = by_sector.get(sec, 0) + float(w)
    return by_sector


if __name__ == '__main__':
    print("Sector breakdown of NIFTY 46 universe:")
    by_sec = {}
    for s, sec in SECTORS.items():
        by_sec.setdefault(sec, []).append(s)
    for sec in sorted(by_sec):
        print(f"\n  {sec} ({len(by_sec[sec])}):")
        for s in by_sec[sec]:
            print(f"    {s}")
    print(f"\nTotal: {len(SECTORS)} stocks across {len(set(SECTORS.values()))} sectors")
