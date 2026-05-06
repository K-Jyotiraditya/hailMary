"""
Hierarchical Risk Parity (HRP) — López de Prado, JPM 2016.

Why HRP over Markowitz mean-variance:
  Markowitz minimizes variance subject to a return target. The math is elegant
  but it's notoriously fragile: the optimizer concentrates weight on assets
  with the lowest sample-variance estimate, even when those estimates are
  noise. Tiny changes in the input matrix produce wildly different weights.

  HRP sidesteps this by clustering assets first, then allocating risk top-down
  through the cluster tree. It NEVER inverts the covariance matrix, so it's
  numerically stable and produces sensible weights even with poor correlation
  estimates. In simulation studies it consistently beats inverse-variance
  weighting and equal-weighting on out-of-sample variance.

Algorithm (3 steps):
  1. CLUSTER: hierarchical clustering on correlation distance d_ij = sqrt(0.5 * (1 - corr_ij))
  2. SERIATE (quasi-diagonalize): reorder rows/cols of cov so similar assets sit next to each other
  3. RECURSIVE BISECTION: split the seriated set in half, allocate inverse-variance
     between halves, recurse into each half.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def correlation_distance(corr: pd.DataFrame) -> np.ndarray:
    """
    Convert correlation matrix into a proper distance metric.
    d_ij = sqrt(0.5 * (1 - rho_ij)) lies in [0, 1] and satisfies triangle inequality.
    """
    return np.sqrt(0.5 * (1 - corr).clip(0, 2))


def quasi_diagonalize(linkage_matrix: np.ndarray) -> list:
    """
    Reorder cluster leaves so similar assets are adjacent in the seriated matrix.
    Returns leaf order (list of indices).
    """
    link = linkage_matrix.astype(int)
    n = link[-1, -1] // 2 + 1  # number of original observations

    # Start with the root cluster (last row of linkage)
    sorted_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, -1]  # initial count

    while sorted_ix.max() >= n:
        # Make space; expand each cluster id into its components
        sorted_ix.index = range(0, sorted_ix.shape[0] * 2, 2)
        df0 = sorted_ix[sorted_ix >= n]
        i = df0.index
        j = df0.values - n
        sorted_ix[i] = link[j, 0]  # left children
        df1 = pd.Series(link[j, 1], index=i + 1)  # right children
        sorted_ix = pd.concat([sorted_ix, df1])
        sorted_ix = sorted_ix.sort_index()
        sorted_ix.index = range(sorted_ix.shape[0])

    return sorted_ix.tolist()


def cluster_variance(cov: pd.DataFrame, items: list) -> float:
    """Inverse-variance portfolio variance for a subset of items."""
    cov_sub = cov.loc[items, items]
    inv_var = 1.0 / np.diag(cov_sub)
    weights = inv_var / inv_var.sum()
    return float(weights @ cov_sub.values @ weights)


def recursive_bisection(cov: pd.DataFrame, sort_ix: list) -> pd.Series:
    """
    Top-down recursive bisection: split sorted items in half, allocate by
    inverse cluster variance, recurse.
    """
    weights = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]

    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left, right = cluster[:mid], cluster[mid:]

            var_left = cluster_variance(cov, left)
            var_right = cluster_variance(cov, right)

            # Inverse-variance weight between the two halves
            alpha = 1 - var_left / (var_left + var_right)

            weights[left] *= alpha
            weights[right] *= 1 - alpha

            new_clusters.extend([left, right])

        clusters = new_clusters

    return weights


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """
    Compute Hierarchical Risk Parity portfolio weights from a returns matrix.

    Args:
        returns: DataFrame with columns = assets, rows = dates of past returns.

    Returns:
        Series of weights summing to 1, indexed by asset name.
    """
    if returns.shape[1] == 1:
        return pd.Series(1.0, index=returns.columns)

    cov = returns.cov()
    corr = returns.corr().clip(-1, 1)

    # 1. Distance matrix and hierarchical clustering
    dist = correlation_distance(corr)
    np.fill_diagonal(dist.values, 0.0)
    link_input = squareform(dist.values, checks=False)
    link = linkage(link_input, method='single')

    # 2. Quasi-diagonalization
    sort_ix_pos = quasi_diagonalize(link)
    sort_ix = corr.index[sort_ix_pos].tolist()

    # 3. Recursive bisection
    weights_by_pos = recursive_bisection(cov.loc[sort_ix, sort_ix], sort_ix)

    # Reindex back to original order
    return weights_by_pos.reindex(returns.columns).fillna(0.0)


def hrp_weights_robust(returns: pd.DataFrame, fallback_to_inv_var: bool = True) -> pd.Series:
    """HRP with safety fallback to inverse-volatility if numerical issues occur."""
    try:
        w = hrp_weights(returns)
        if w.isna().any() or (w < 0).any() or w.sum() <= 0:
            raise ValueError("HRP returned invalid weights")
        return w / w.sum()
    except Exception:
        if not fallback_to_inv_var:
            raise
        # Fallback: inverse-volatility weights
        vol = returns.std()
        inv_vol = 1.0 / vol.replace(0, np.nan)
        weights = inv_vol / inv_vol.sum()
        return weights.fillna(1.0 / len(returns.columns))


if __name__ == '__main__':
    # Smoke test on the 6 tradeable stocks
    tradeable = ['AXISBANK', 'ITC', 'NTPC', 'RELIANCE', 'SBILIFE', 'TCS']
    returns_dict = {}
    for s in tradeable:
        df = pd.read_parquet(f'data/processed/{s}.parquet')
        returns_dict[s] = df['Close'].pct_change()
    returns = pd.DataFrame(returns_dict).dropna()

    # Use last 60 days for HRP
    recent = returns.tail(60)
    weights = hrp_weights_robust(recent)
    print("HRP weights on the 6 tradeable stocks (last 60 days):")
    print(weights.sort_values(ascending=False).round(4).to_string())
    print(f"\nSum: {weights.sum():.4f} (should be 1.0)")
    print(f"Effective N (1/sum_w^2): {1/(weights**2).sum():.2f}")
