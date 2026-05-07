"""
Combined Universe Screener: S&P 500 + S&P MidCap 400.

Weekly: screens ~700 stocks down to 50-80 high-quality candidates using:
  - Momentum: 200-day MA filter + 12-1 month return top quartile
  - Liquidity: avg daily dollar volume > $50M (lower bar for mid-caps)
  - Excludes penny stocks and tickers with insufficient history

Mid-caps are included because they have LESS analyst coverage than
mega-caps, which means information asymmetry exists — our signals can
actually have predictive power there.

Results cached to data/universe_cache.json for 7 days.
Import get_universe() instead of WATCHLIST when using the Phase 2 pipeline.
"""
import json
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

CACHE_PATH = Path("data/universe_cache.json")
CACHE_TTL_DAYS = 7

# S&P 500 ticker list (static snapshot — refreshed manually or via data pull)
SP500_TICKERS = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM","ALB",
    "ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AAL","AEP",
    "AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON","APA","AAPL","AMAT",
    "APTV","ACGL","ANET","AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR",
    "BALL","BAC","BBWI","BAX","BDX","BRK-B","BBY","BIO","TECH","BIIB","BLK","BX","BA","BCR",
    "BMY","AVGO","BR","BRO","BF-B","BLDR","BG","CDNS","CZR","CPT","CPB","COF","CAH","KMX",
    "CCL","CARR","CTLT","CAT","CBOE","CBRE","CDW","CE","COR","CNC","CNP","CF","CHTR","CVX",
    "CMG","CB","CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME","CMS","KO","CTSH","CL",
    "CMCSA","CMA","CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CTVA","CSGP","COST","CTRA",
    "CCI","CSX","CMI","CVS","DHI","DHR","DRI","DVA","DE","DAL","XRAY","DVN","DXCM","FANG",
    "DLR","DFS","DG","DLTR","D","DPZ","DOV","DOW","DHR","DTE","DUK","DD","EMN","ETN","EBAY",
    "ECL","EIX","EW","EA","ELV","LLY","EMR","ENPH","ETR","EOG","EPAM","EQT","EFX","EQIX",
    "EQR","ESS","EL","ETSY","EG","EVRG","ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS",
    "FICO","FAST","FRT","FDX","FIS","FITB","FE","FRC","FLT","FMC","F","FTNT","FTV","FOXA",
    "FOX","BEN","FCX","GRMN","IT","GEHC","GEN","GNRC","GD","GE","GIS","GM","GPC","GILD",
    "GL","GPN","GS","HAL","HIG","HAS","HCA","PEAK","HSIC","HSY","HES","HPE","HLT","HOLX",
    "HD","HON","HRL","HST","HWM","HPQ","HUM","HBAN","HII","IBM","IEX","IDXX","ITW","ILMN",
    "INCY","IR","PODD","INTC","ICE","IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV","IRM",
    "JBHT","JKHY","J","JNJ","JCI","JPM","JNPR","K","KEYS","KMB","KIM","KMI","KLAC","KHC",
    "KR","LHX","LH","LRCX","LW","LVS","LDOS","LEN","LNC","LIN","LYV","LKQ","LMT","L","LOW",
    "LULU","LYB","MTB","MRO","MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH","MKC","MCD",
    "MCK","MDT","MRK","META","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA","MHK","MOH",
    "TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP","NFLX","NWL",
    "NEM","NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE","NVDA",
    "NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OGN","OTIS","PCAR","PKG","PARA",
    "PH","PAYX","PAYC","PYPL","PNR","PEP","PKI","PFE","PCG","PM","PSX","PNW","PXD","PNC",
    "POOL","PPG","PPL","PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM","QRVO","PWR",
    "QCOM","DGX","RL","RJF","RTX","O","REG","REGN","RF","RSG","RMD","RVTY","RHI","ROK","ROL",
    "ROP","ROST","RCL","SPGI","CRM","SBAC","SLB","STX","SEE","SRE","NOW","SHW","SPG","SWKS",
    "SJM","SNA","SEDG","SO","LUV","SWK","SBUX","STT","STLD","STE","SYK","SYF","SNPS","SYY",
    "TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL","TDY","TFX","TER","TSLA","TXN","TXT","TMO",
    "TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL","TSN","USB","UDR","ULTA","UNP","UAL",
    "UPS","URI","UNH","UHS","VLO","VTR","VRSN","VRSK","VZ","VRTX","VFC","VTRS","VICI","V",
    "VMC","WAB","WBA","WMT","WBD","WM","WAT","WEC","WFC","WELL","WST","WDC","WRK","WY","WHR",
    "WMB","WTW","GWW","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZION","ZTS","AAPL","MSFT",
]

# De-duplicate
SP500_TICKERS = list(dict.fromkeys(SP500_TICKERS))

# S&P MidCap 400 — curated liquid names across sectors
# Less analyst coverage = more information asymmetry = real ML edge
MIDCAP_TICKERS = [
    # Technology / Software
    "PSTG", "GTLB", "WK", "PCTY", "FOUR", "QLYS", "ALTR", "NEWR",
    "JAMF", "YEXT", "HUBS", "DOMO", "BRZE", "CFLT", "TASK", "BILL",
    # Semiconductors (mid)
    "FORM", "DIOD", "LSCC", "RMBS", "SITM", "SMTC", "CEVA", "AMBA",
    # Financials / Regional Banks
    "FHN", "IBKR", "EWBC", "SNV", "WTFC", "CFR", "COLB", "FNB",
    "WAL", "BANR", "BOKF", "CADE", "FFIN", "INDB", "SBCF", "PACW",
    "NBTB", "WSFS", "TCBI", "EFSC",
    # Insurance / Finance Services
    "GSHD", "RYAN", "AON", "EG", "RLI", "SIGI",
    # Healthcare / Biotech mid
    "EXAS", "INSP", "RDNT", "OMCL", "PDCO", "ACAD", "HIMS", "TMDX",
    "DOCS", "PRVA", "SGRY", "TXG", "ITCI", "ACLS", "AMED", "HALO",
    "NVCR", "PTGX", "RYTM", "SWTX", "VKTX",
    # Consumer Discretionary
    "RH", "WING", "BOOT", "SFM", "PFGC", "CHUY", "EYE", "ONON",
    "SHAK", "MODG", "GENI", "JACK", "CAKE", "BJ", "GPI", "LAD",
    # Consumer Staples (mid)
    "CHEF", "INGR", "SPTN", "UNFI", "USFD", "CASY", "GO",
    # Industrials
    "FLR", "KNX", "SAIA", "GXO", "HUBG", "MATX", "GATX",
    "WSC", "LSTR", "WERN", "AL", "CAR", "TRN", "AIT", "ESE",
    "KTOS", "MOOG", "SPXC", "TDW",
    # Energy (mid)
    "CIVI", "MGY", "MTDR", "SM", "VTLE", "DINO", "CHRD", "DEN",
    "MUR", "OVV", "REI", "TALO", "PARR",
    # Materials
    "CLF", "CRS", "HCC", "KALU", "NEU", "SLVM", "MP", "TREX",
    "UFPI", "SUM", "STLD", "OMG", "HWKN",
    # Real Estate
    "BNL", "CDP", "HIW", "IRT", "KRG", "LXP", "PDM", "RLJ",
    "SKT", "STAG", "SAFE", "PLYM", "RPT",
    # Utilities (mid)
    "ALE", "AVA", "MGEE", "NWE", "OGS", "OTTR", "SJW", "SWX",
    # Communication / Media
    "CARG", "DLB", "MSGS", "SBGI", "CARS", "IDT", "IACI",
]

# De-duplicate mid-caps
MIDCAP_TICKERS = list(dict.fromkeys(MIDCAP_TICKERS))

# Combined universe
FULL_UNIVERSE = list(dict.fromkeys(SP500_TICKERS + MIDCAP_TICKERS))

MIN_AVG_DOLLAR_VOL = 50e6    # $50M/day (lower bar covers mid-caps)
TOP_MOMENTUM_PCT   = 0.50    # Keep top 50% by 12-1 month momentum
MAX_CANDIDATES     = 80
MIN_CANDIDATES     = 20


def _load_cache() -> list | None:
    if not CACHE_PATH.exists():
        return None
    try:
        with open(CACHE_PATH) as f:
            data = json.load(f)
        cached_date = datetime.fromisoformat(data["date"])
        if datetime.now() - cached_date < timedelta(days=CACHE_TTL_DAYS):
            return data["tickers"]
    except Exception:
        pass
    return None


def _save_cache(tickers: list):
    CACHE_PATH.parent.mkdir(exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump({"date": datetime.now().isoformat(), "tickers": tickers}, f)


def screen_universe(force_refresh: bool = False) -> list:
    """
    Screen S&P 500 to 50-80 high-quality candidates.
    Returns sorted list of ticker strings.
    Cached for 7 days; force_refresh=True bypasses cache.
    """
    if not force_refresh:
        cached = _load_cache()
        if cached:
            print(f"  [Universe] Using cached universe ({len(cached)} stocks, refreshes weekly)")
            return cached

    print(f"  [Universe] Screening {len(FULL_UNIVERSE)} stocks (S&P 500 + MidCap 400)...")

    try:
        raw = yf.download(
            FULL_UNIVERSE, period="14mo", auto_adjust=True,
            progress=False, threads=True
        )
    except Exception as e:
        print(f"  [Universe] Download failed: {e} — falling back to static watchlist")
        from config.watchlist import WATCHLIST
        return WATCHLIST

    if not isinstance(raw.columns, pd.MultiIndex):
        from config.watchlist import WATCHLIST
        return WATCHLIST

    close = raw["Close"].dropna(axis=1, how="all")
    volume = raw["Volume"].dropna(axis=1, how="all") if "Volume" in raw.columns.get_level_values(0) else pd.DataFrame()

    candidates = []
    for ticker in close.columns:
        c = close[ticker].dropna()
        if len(c) < 200:
            continue

        # 200-day MA filter
        sma200 = c.rolling(200).mean().iloc[-1]
        last_price = c.iloc[-1]
        if last_price < sma200:
            continue  # below 200-day MA — skip

        # Liquidity: avg dollar volume over last 63 days
        if ticker in volume.columns:
            v = volume[ticker].dropna()
            avg_dv = float((v * c.reindex(v.index)).tail(63).mean())
            if avg_dv < MIN_AVG_DOLLAR_VOL:
                continue

        # 12-1 month momentum (skip first month to avoid short-term reversal)
        if len(c) < 252:
            continue
        momentum = float(c.iloc[-21] / c.iloc[-252] - 1) if len(c) >= 252 else 0.0

        candidates.append({"ticker": ticker, "momentum": momentum})

    if not candidates:
        from config.watchlist import WATCHLIST
        return WATCHLIST

    df = pd.DataFrame(candidates).sort_values("momentum", ascending=False)

    # Keep top 50% by momentum, cap at MAX_CANDIDATES
    top_n = max(MIN_CANDIDATES, min(MAX_CANDIDATES, int(len(df) * TOP_MOMENTUM_PCT)))
    selected = df.head(top_n)["ticker"].tolist()

    print(f"  [Universe] Selected {len(selected)} stocks (from {len(candidates)} passing MA+liquidity filters)")
    _save_cache(selected)
    return selected


def get_universe(force_refresh: bool = False) -> list:
    """Public entry point — returns screened ticker list."""
    return screen_universe(force_refresh=force_refresh)
