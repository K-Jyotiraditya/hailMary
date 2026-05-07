"""
Hierarchical Risk Parity (HRP) — Lopez de Prado (2016).

Replaces naive equal-weighting or LLM-guessed weights with a
mathematically principled, correlation-aware allocation that:
  - Clusters assets by correlation (Ward linkage)
  - Quasi-diagonalizes the covariance matrix
  - Allocates via recursive bisection (inverse-variance within clusters)

This produces diversified portfolios that are more robust to estimation
error than mean-variance optimization.

Usage:
    weights = compute_hrp_weights(tickers, lookback_days=252)
    # Returns dict {ticker: weight} summing to 1.0
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def _corr_to_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Convert correlation matrix to distance matrix: d = sqrt(0.5*(1-rho))."""
    return np.sqrt(0.5 * (1.0 - corr)).clip(0, 1)


def _get_quasi_diag(link: np.ndarray) -> list:
    """
    Quasi-diagonalize the linkage matrix.
    Returns sorted leaf indices (original asset indices).
    """
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]  # total leaves

    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]  # find clusters
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix = sort_ix.reset_index(drop=True)

    return sort_ix.tolist()


def _get_cluster_var(cov: pd.DataFrame, c_items: list) -> float:
    """Return the variance of the inverse-variance portfolio on the cluster."""
    cov_slice = cov.iloc[c_items, c_items]
    ivp = 1.0 / np.diag(cov_slice.values)
    ivp /= ivp.sum()
    return float(ivp @ cov_slice.values @ ivp)


def _hrp_recursive_bisection(cov: pd.DataFrame, sort_ix: list) -> pd.Series:
    """Allocate weights via recursive bisection."""
    w = pd.Series(1.0, index=cov.index)
    c_items = [sort_ix]  # start with the full sorted list

    while len(c_items) > 0:
        c_items = [
            i[j:k]
            for i in c_items
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]
        for i in range(0, len(c_items), 2):
            c_items0 = c_items[i]
            c_items1 = c_items[i + 1]
            alpha = 1.0 - _get_cluster_var(cov, c_items0) / (
                _get_cluster_var(cov, c_items0) + _get_cluster_var(cov, c_items1)
            )
            w.iloc[c_items0] *= alpha
            w.iloc[c_items1] *= (1 - alpha)

    return w


def compute_hrp_weights(tickers: list, lookback_days: int = 252,
                        close_df: pd.DataFrame | None = None) -> dict:
    """
    Compute HRP weights for a list of tickers.

    Args:
        tickers: list of ticker strings
        lookback_days: days of history used for covariance estimation
        close_df: pre-downloaded close DataFrame (avoids re-download if available)

    Returns:
        dict {ticker: weight} summing to 1.0, or equal weights on failure.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    equal = {t: 1.0 / len(tickers) for t in tickers}
    if len(tickers) < 2:
        return equal

    try:
        if close_df is not None:
            prices = close_df[tickers].dropna(axis=1, how="all").tail(lookback_days + 5)
        else:
            raw = yf.download(tickers, period="1y", auto_adjust=True,
                              progress=False, threads=False)
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"]
            else:
                return equal

        # Keep only tickers with enough data
        prices = prices.dropna(axis=1, thresh=int(lookback_days * 0.8))
        valid_tickers = [t for t in tickers if t in prices.columns]
        if len(valid_tickers) < 2:
            return equal
        prices = prices[valid_tickers].tail(lookback_days)

        returns = prices.pct_change().dropna()
        if len(returns) < 30:
            return equal

        cov = returns.cov()
        corr = returns.corr()

        # Replace NaN/inf with 0 correlation
        corr_arr = corr.fillna(0).clip(-1, 1).to_numpy(copy=True, dtype=float)
        np.fill_diagonal(corr_arr, 1.0)
        corr = pd.DataFrame(corr_arr, index=corr.index, columns=corr.columns)

        dist = _corr_to_distance(corr)
        dist_sq = squareform(dist.values, checks=False).copy()
        dist_sq = np.clip(dist_sq, 0, None)

        link = linkage(dist_sq, method="ward")
        sort_ix = _get_quasi_diag(link)

        # Reorder cov to match quasi-diagonalized order
        sort_tickers = [valid_tickers[i] for i in sort_ix]
        cov_sorted = cov.loc[sort_tickers, sort_tickers]

        w = _hrp_recursive_bisection(cov_sorted, list(range(len(sort_tickers))))
        w = w / w.sum()  # normalize to 1

        result = {t: float(w[t]) for t in sort_tickers}

        # Fill missing tickers with 0 (they didn't pass the data filter)
        for t in tickers:
            if t not in result:
                result[t] = 0.0

        return result

    except Exception as e:
        print(f"  [HRP] Failed ({e}), using equal weights")
        return equal
