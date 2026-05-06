"""
Combinatorial Purged Cross-Validation (CPCV)

Overrides standard Walk-Forward by chunking time into N groups and testing 
on combinations of k groups, generating multiple (N choose k) backtest paths.

Includes Deflated Sharpe Ratio (DSR) (Bailey & Lopez de Prado 2014) to adjust 
for selection bias across the multiple generated paths.
"""
import numpy as np
import pandas as pd
import scipy.stats as ss
import math
from itertools import combinations
from typing import Iterator, Tuple, List

class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation.
    Divides time series into N chunks. Uses Combinations to select k chunks 
    as the test set. 
    Embargo/Purge must >= Label Horizon.
    """
    def __init__(self, n_groups: int = 10, k_test_groups: int = 2, 
                 purge_days: int = 5, embargo_days: int = 5):
        self.n = n_groups
        self.k = k_test_groups
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if isinstance(X.index, pd.MultiIndex):
            dates = pd.Series(X.index.get_level_values('date')).dt.date.sort_values().unique()
            X_dates = pd.to_datetime(X.index.get_level_values('date')).dt.date
        else:
            dates = pd.Series(X.index).dt.date.sort_values().unique()
            X_dates = pd.to_datetime(X.index).dt.date

        # 1. Split timeline into N groups
        groups = np.array_split(dates, self.n)
        
        # 2. combinations of k groups for testing
        test_combinations = list(combinations(range(self.n), self.k))
        
        for test_idx_list in test_combinations:
            test_dates = pd.Series(np.concatenate([groups[i] for i in test_idx_list]))
            
            # Determine Train chunks (everything else)
            train_dates = pd.Series(np.concatenate([groups[i] for i in range(self.n) if i not in test_idx_list]))
            
            # 3. Purging and Embargoing
            # For EACH test group chunk, apply the purge before and embargo after
            valid_train = train_dates.copy()
            for t_idx in test_idx_list:
                chunk = groups[t_idx]
                chunk_start = pd.Timestamp(chunk.min())
                chunk_end = pd.Timestamp(chunk.max())
                
                purge_cutoff = chunk_start - pd.Timedelta(days=self.purge_days)
                embargo_cutoff = chunk_end + pd.Timedelta(days=self.embargo_days)
                
                # Filter out the invalid dates from valid_train
                valid_train = valid_train[
                    (pd.to_datetime(valid_train) <= purge_cutoff) | 
                    (pd.to_datetime(valid_train) >= embargo_cutoff)
                ]
            
            train_mask = X_dates.isin(valid_train)
            test_mask = X_dates.isin(test_dates)
            
            yield np.where(train_mask)[0], np.where(test_mask)[0]

def deflated_sharpe_ratio(empirical_sharpes: List[float], returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Computes the Deflated Sharpe Ratio (DSR).
    
    Args:
        empirical_sharpes: List of Out-of-Sample Sharpes from CPCV paths
        returns: The daily returns of the *chosen* path
        risk_free_rate: Annualized risk-free rate
    Returns:
        Probability that the max Sharpe is statistically significant.
    """
    n_trials = len(empirical_sharpes)
    if n_trials < 2:
        return np.nan
        
    mean_sr = np.mean(empirical_sharpes)
    std_sr = np.std(empirical_sharpes)
    
    # 1. Calculate Expected Maximum Sharpe Ratio (SR_star)
    gamma_em = 0.57721566  # Euler-Mascheroni constant
    prob = 1.0 - (1.0 / n_trials)
    prob_e = 1.0 - (1.0 / (n_trials * math.e))
    max_z = (1 - gamma_em) * ss.norm.ppf(prob) + gamma_em * ss.norm.ppf(prob_e)
    
    sr_star = mean_sr + std_sr * max_z
    
    # 2. Calculate DSR using chosen path's properties
    chosen_sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    t = len(returns)
    skew = ss.skew(returns)
    kurtosis = ss.kurtosis(returns) # Fisher (excess)
    
    variance_of_sharpe = (1 - skew * chosen_sharpe + (kurtosis / 4) * (chosen_sharpe ** 2)) / (t - 1)
    
    if variance_of_sharpe <= 0:
        return 0.0
        
    # Prob(SR_test > SR_star)
    dsr_stat = (chosen_sharpe - sr_star) / np.sqrt(variance_of_sharpe)
    dsr_prob = ss.norm.cdf(dsr_stat)
    
    return float(dsr_prob)
