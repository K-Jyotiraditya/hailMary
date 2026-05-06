"""
Data Loader - Unified interface for loading OHLCV data.
Handles yfinance CSV format quirks and provides clean parquet storage.
"""
import pandas as pd
from pathlib import Path


def load_yfinance_csv(file_path: str) -> pd.DataFrame:
    """Load a yfinance-exported CSV with multi-row header."""
    df = pd.read_csv(file_path, skiprows=[1, 2], header=0)
    df = df.rename(columns={'Price': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df


def load_all_stocks(data_dir: str = 'data/raw', file_format: str = 'csv') -> dict:
    """Load all stocks from a directory."""
    data_path = Path(data_dir)

    if file_format == 'csv':
        files = list(data_path.glob('*.csv'))
        loader = load_yfinance_csv
    elif file_format == 'parquet':
        files = list(data_path.glob('*.parquet'))
        loader = pd.read_parquet
    else:
        raise ValueError(f"Unknown format: {file_format}")

    all_data = {}
    for file_path in sorted(files):
        symbol = file_path.stem
        all_data[symbol] = loader(str(file_path))

    return all_data


def convert_csv_to_parquet(input_dir: str = 'data/raw', output_dir: str = 'data/processed'):
    """Convert all yfinance CSVs to clean parquet files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    csv_files = sorted(Path(input_dir).glob('*.csv'))
    print(f"Converting {len(csv_files)} CSV files to parquet...")

    for csv_file in csv_files:
        symbol = csv_file.stem
        df = load_yfinance_csv(str(csv_file))
        out_path = Path(output_dir) / f"{symbol}.parquet"
        df.to_parquet(out_path)
        print(f"  {symbol:15} | {len(df):6d} rows | {df.index.min().date()} to {df.index.max().date()}")

    print(f"\nDone. Parquet files saved to {output_dir}")


if __name__ == '__main__':
    convert_csv_to_parquet()
