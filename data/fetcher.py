"""
Data Fetcher - Fetch NIFTY 50 data from yfinance with error handling.
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

import os

import os
import requests

def get_foreign_universe(market: str) -> list:
    """
    Fetches international Constituents from Wikipedia.
    """
    try:
        if market == 'JAPAN':
            url = 'https://en.wikipedia.org/wiki/Nikkei_225'
            res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            dfs = pd.read_html(res.text)
            for df in dfs:
                # Japanese stocks use 4-digit codes on Wikipedia
                if any('Code' in c or 'Ticker' in c for c in df.columns):
                    col = next(c for c in df.columns if 'Code' in c or 'Ticker' in c)
                    tickers = df[col].astype(str) + '.T'
                    return tickers.tolist()
                    
        elif market == 'UK':
            url = 'https://en.wikipedia.org/wiki/FTSE_100_Index'
            res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            dfs = pd.read_html(res.text)
            for df in dfs:
                if 'Ticker' in df.columns:
                    tickers = df['Ticker'].astype(str) + '.L'
                    return tickers.tolist()

    except Exception as e:
        print(f"Failed to scrape Wikipedia {market}: {e}")
        
    print(f"WARNING: Wikipedia scraper failed. Falling back to static {market} list.")
    # Fallback to extremely basic proxies for verification
    if market == 'JAPAN':
        return ['7203.T', '9984.T', '8306.T', '6758.T', '6861.T', '9432.T', '8035.T']
    elif market == 'UK':
        return ['SHEL.L', 'AZN.L', 'HSBA.L', 'ULVR.L', 'BP.L', 'RIO.L', 'GSK.L', 'DGE.L']
    
    return []

# Fallback symbols in case of issues
SYMBOL_MAPPING = {}


def fetch_stock_data(symbol: str, start_date: str, end_date: str, max_retries: int = 3) -> pd.DataFrame:
    """
    Fetch OHLCV data for a stock from yfinance with retries.

    Args:
        symbol: Ticker symbol (e.g., 'RELIANCE.NS')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_retries: Maximum retry attempts

    Returns:
        DataFrame with OHLCV data, or empty DataFrame if failed
    """
    for attempt in range(max_retries):
        try:
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                timeout=10,
                multi_level_index=False
            )

            if data.empty:
                if attempt < max_retries - 1:
                    continue
                return pd.DataFrame()

            # Clean column names (remove spaces)
            data.columns = [col.strip() for col in data.columns]
            return data

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to fetch {symbol}: {str(e)}")
                return pd.DataFrame()
            continue

    return pd.DataFrame()


def fetch_nifty_20(start_date: str = None, end_date: str = None, save_path: str = None) -> dict:
    """
    Fetch OHLCV data for all NIFTY 20 stocks.

    Args:
        start_date: Start date (default: 10 years ago)
        end_date: End date (default: today)
        save_path: Path to save data (pickle format)

    Returns:
        Dictionary {symbol: DataFrame}
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')

    import sys
    market = 'JAPAN'
    if len(sys.argv) > 1:
        market = sys.argv[1].upper()
        
    all_data = {}

    universe = get_foreign_universe(market)
    
    benchmark = '^N225' if market == 'JAPAN' else '^FTSE'
    if benchmark not in universe:
        universe.append(benchmark) # Add benchmark index for regime filter
        
    print(f"\nFetching {market} Equities data from {start_date} to {end_date}")
    print(f"Symbols: {len(universe)}")

    for symbol in tqdm(universe, desc="Downloading"):
        data = fetch_stock_data(symbol, start_date, end_date)

        if not data.empty:
            all_data[symbol] = data
            print(f"  ✓ {symbol}: {len(data)} records")
        else:
            print(f"  ✗ {symbol}: Failed")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(all_data, f)
        print(f"\nData saved to {save_path}")

    return all_data


def load_data(path: str) -> dict:
    """Load previously fetched data from pickle."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_data_parquet(all_data: dict, output_dir: str = 'data/raw'):
    """Save all data to parquet files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for symbol, df in all_data.items():
        if symbol in ['^GSPC', '^N225', '^FTSE']:
            filename = f"{output_dir}/BENCHMARK_INDEX.parquet"
        else:
            filename = f"{output_dir}/{symbol.replace('.', '-')}.parquet"
        df.to_parquet(filename)
        print(f"Saved {symbol} to {filename}")


def load_data_parquet(symbol: str, data_dir: str = 'data/raw') -> pd.DataFrame:
    """Load a stock's data from parquet."""
    filename = f"{data_dir}/{symbol.replace('.NS', '').replace('.', '_')}.parquet"
    return pd.read_parquet(filename)


if __name__ == '__main__':
    # Fetch and save data
    all_data = fetch_nifty_20(save_path='data/nifty_20.pkl')
    save_data_parquet(all_data)
