"""
Fetch and clean NIFTY 20 data from yfinance.
Usage: python scripts/01_fetch_data.py
"""
import sys
sys.path.insert(0, '/c/Users/yashj/Desktop/HailMary')

from data.fetcher import fetch_nifty_20, save_data_parquet
from data.cleaner import clean_all_data
from config.settings import DATA_CONFIG
import pandas as pd

if __name__ == '__main__':
    print("=" * 60)
    print("STEP 1: FETCH NIFTY 20 DATA")
    print("=" * 60)

    # Fetch data
    print(f"\nFetching data from {DATA_CONFIG['start_date']} to present...")
    all_data = fetch_nifty_20(
        start_date=DATA_CONFIG['start_date'],
        end_date=DATA_CONFIG['end_date'],
        save_path='data/nifty_20.pkl'
    )

    print(f"\nFetched {len(all_data)} symbols")

    # Show summary
    print("\nData Summary:")
    for symbol, df in all_data.items():
        print(f"  {symbol:20} | {len(df):6d} records | {df.index.min().date()} to {df.index.max().date()}")

    # Clean data
    print("\n" + "=" * 60)
    print("STEP 2: CLEAN DATA")
    print("=" * 60)

    cleaned_data = clean_all_data(all_data)

    # Save to parquet
    print("\n" + "=" * 60)
    print("STEP 3: SAVE TO PARQUET")
    print("=" * 60)

    save_data_parquet(cleaned_data, output_dir='data/processed')

    print("\n" + "=" * 60)
    print("✓ Data fetch and clean complete!")
    print("=" * 60)
