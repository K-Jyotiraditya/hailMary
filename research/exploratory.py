"""
Exploratory Data Analysis - Check stationarity, correlations, volatility clustering.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def adf_test(series: pd.Series, name: str = '') -> dict:
    """
    Augmented Dickey-Fuller test for stationarity.

    Returns:
        dict with test_statistic, p_value, is_stationary
    """
    from scipy.stats import adfuller

    result = adfuller(series.dropna(), autolag='AIC')

    return {
        'name': name,
        'statistic': result[0],
        'p_value': result[1],
        'is_stationary': result[1] < 0.05,
    }


def check_autocorrelation(series: pd.Series, lags: int = 20) -> dict:
    """Check autocorrelation in returns."""
    from scipy.stats import spearmanr

    acf_values = [series.autocorr(lag=i) for i in range(1, lags + 1)]

    return {
        'mean_acf': np.mean(np.abs(acf_values)),
        'max_acf': np.max(np.abs(acf_values)),
        'is_autocorrelated': np.max(np.abs(acf_values)) > 0.2,
    }


def check_volatility_clustering(returns: pd.Series, threshold: float = 0.02) -> dict:
    """
    Check for volatility clustering using ARCH effects.

    High volatility clustering suggests GARCH-like behavior.
    """
    squared_returns = returns ** 2
    acf_squared = [squared_returns.autocorr(lag=i) for i in range(1, 21)]

    mean_acf_squared = np.mean(np.abs(acf_squared))

    return {
        'mean_acf_squared': mean_acf_squared,
        'has_clustering': mean_acf_squared > 0.1,
        'acf_values': acf_squared,
    }


def analyze_data(data_dir: str = 'data/raw') -> dict:
    """
    Comprehensive EDA on all fetched data.

    Returns:
        Dictionary with analysis results
    """
    csv_files = list(Path(data_dir).glob('*.csv'))
    print(f"Found {len(csv_files)} CSV files\n")

    analysis_results = {}

    for file_path in sorted(csv_files):
        symbol = file_path.stem
        print(f"Analyzing {symbol}...")

        try:
            # Load data
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            df = df.sort_index()

            # Basic stats
            n_records = len(df)
            date_range = f"{df.index.min().date()} to {df.index.max().date()}"

            # Returns analysis
            returns = df['Close'].pct_change().dropna()

            # Stationarity test (returns should be stationary)
            adf_result = adf_test(returns, name=f"{symbol}_returns")

            # Autocorrelation
            autocorr_result = check_autocorrelation(returns)

            # Volatility clustering
            vol_cluster_result = check_volatility_clustering(returns)

            # Basic stats
            ret_mean = returns.mean() * 252  # Annualized
            ret_std = returns.std() * np.sqrt(252)  # Annualized
            sharpe = ret_mean / ret_std if ret_std > 0 else 0
            skew = returns.skew()
            kurt = returns.kurtosis()

            analysis_results[symbol] = {
                'n_records': n_records,
                'date_range': date_range,
                'return_mean_annual': ret_mean,
                'return_std_annual': ret_std,
                'sharpe_ratio': sharpe,
                'skewness': skew,
                'kurtosis': kurt,
                'is_stationary': adf_result['is_stationary'],
                'adf_p_value': adf_result['p_value'],
                'autocorrelation': autocorr_result['mean_acf'],
                'vol_clustering': vol_cluster_result['mean_acf_squared'],
            }

            print(f"  Records: {n_records}, Stationary: {adf_result['is_stationary']}")

        except Exception as e:
            print(f"  ERROR: {str(e)}")

    return analysis_results


def print_analysis_summary(results: dict):
    """Print summary of analysis."""
    print("\n" + "=" * 80)
    print("EDA SUMMARY")
    print("=" * 80)

    df_summary = pd.DataFrame(results).T
    df_summary = df_summary.round(4)

    print(f"\nStatistics for {len(results)} stocks:")
    print(df_summary.to_string())

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    stationary_count = df_summary['is_stationary'].sum()
    print(f"Stationary Returns: {stationary_count}/{len(results)} stocks")

    mean_sharpe = df_summary['sharpe_ratio'].mean()
    print(f"Mean Sharpe Ratio: {mean_sharpe:.4f}")

    mean_vol_cluster = df_summary['vol_clustering'].mean()
    print(f"Mean Volatility Clustering (ACF squared): {mean_vol_cluster:.4f}")

    print("\nInterpretation:")
    if stationary_count == len(results):
        print("  ✓ All returns are stationary (good for ML)")
    else:
        print(f"  ⚠ Only {stationary_count} stocks have stationary returns")

    if mean_vol_cluster > 0.1:
        print("  ✓ Strong volatility clustering detected (GARCH effects)")
    else:
        print("  ⚠ Weak volatility clustering (less autocorrelation)")


def compute_correlation_matrix(data_dir: str = 'data/raw') -> pd.DataFrame:
    """Compute correlation matrix across all stocks."""
    csv_files = list(Path(data_dir).glob('*.csv'))

    all_returns = {}

    for file_path in sorted(csv_files):
        symbol = file_path.stem
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            returns = df['Close'].pct_change()
            all_returns[symbol] = returns
        except:
            pass

    returns_df = pd.DataFrame(all_returns)
    correlation_matrix = returns_df.corr()

    print("\nCorrelation Matrix (Top 5 pairs):")
    print(correlation_matrix)

    return correlation_matrix


if __name__ == '__main__':
    print("=" * 80)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80 + "\n")

    # Analyze all data
    results = analyze_data(data_dir='data/raw')

    # Print summary
    print_analysis_summary(results)

    # Correlation matrix
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    corr_matrix = compute_correlation_matrix(data_dir='data/raw')

    print("\n✓ EDA complete!")
