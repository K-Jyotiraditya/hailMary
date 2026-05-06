"""
Probability of Backtest Overfitting (PBO) — Bailey, Borwein, López de Prado, Zhu (2014).

THE PROBLEM PBO SOLVES:
  When you try N variants of a strategy (different thresholds, lookbacks, etc.)
  and report the BEST one, you've selected on noise. The "best" Sharpe is biased
  upward by random luck. PBO quantifies that bias.

THE TEST:
  Stack the returns of all N strategy variants into a matrix R [T x N].
  Split T rows into S non-overlapping submatrices (typically S=16).
  For each combination of S/2 chosen as TRAIN and the other S/2 as TEST:
    - Find which strategy ranks #1 in TRAIN (in-sample winner)
    - Compute its TEST rank
    - PBO event = "the IS winner ranks below median in OOS"
  PBO = fraction of combinations where this event occurs.

INTERPRETATION:
  PBO < 0.5: low overfitting — the in-sample winner generally stays good.
  PBO > 0.5: severe overfitting — picking the IS-best is worse than random.
"""
import numpy as np
import pandas as pd
from itertools import combinations


def split_returns_matrix(returns_df: pd.DataFrame, S: int = 16) -> list:
    """
    Split T rows of returns into S non-overlapping submatrices.

    Returns:
        List of S DataFrames, each [T/S x N].
    """
    T = len(returns_df)
    chunk_size = T // S
    chunks = []
    for k in range(S):
        start = k * chunk_size
        end = (k + 1) * chunk_size if k < S - 1 else T
        chunks.append(returns_df.iloc[start:end])
    return chunks


def annualized_sharpe(returns: pd.Series) -> float:
    if len(returns) < 5 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))


def compute_pbo(returns_df: pd.DataFrame, S: int = 16,
                metric_fn=annualized_sharpe) -> dict:
    """
    Compute PBO for a panel of strategy returns.

    Args:
        returns_df: DataFrame, rows = dates, columns = strategy variants.
                    Each column should contain daily returns of one variant.
        S: Number of non-overlapping submatrices (must be even).
        metric_fn: Performance metric per strategy (default: annualized Sharpe).

    Returns:
        Dict with PBO, logit-loss distribution, IS/OOS rank stats.
    """
    if S % 2 != 0:
        raise ValueError("S must be even for the half-and-half split")

    chunks = split_returns_matrix(returns_df, S)
    n_strategies = returns_df.shape[1]
    chunk_indices = list(range(S))

    logit_losses = []
    is_winner_oos_ranks = []
    oos_relative_ranks = []  # Rank of IS-winner in OOS, normalized to [0,1]

    # All combinations of S/2 chunks chosen as IS, rest as OOS
    half = S // 2
    combos = list(combinations(chunk_indices, half))

    for is_combo in combos:
        oos_combo = [i for i in chunk_indices if i not in is_combo]

        is_returns = pd.concat([chunks[i] for i in is_combo], axis=0)
        oos_returns = pd.concat([chunks[i] for i in oos_combo], axis=0)

        is_metrics  = is_returns.apply(metric_fn).values
        oos_metrics = oos_returns.apply(metric_fn).values

        # IS winner
        winner_idx = int(np.argmax(is_metrics))

        # OOS rank of the winner (rank from low to high, 1-based)
        oos_ranks = pd.Series(oos_metrics).rank(method='average')
        winner_oos_rank = oos_ranks.iloc[winner_idx]
        relative_rank = (winner_oos_rank - 1) / (n_strategies - 1) if n_strategies > 1 else 0.5

        is_winner_oos_ranks.append(winner_oos_rank)
        oos_relative_ranks.append(relative_rank)

        # Logit of relative rank
        eps = 1e-9
        rr = np.clip(relative_rank, eps, 1 - eps)
        logit = np.log(rr / (1 - rr))
        logit_losses.append(logit)

    logit_arr = np.array(logit_losses)
    rel_ranks = np.array(oos_relative_ranks)

    # PBO = P(IS-winner ranks below median in OOS) = P(logit < 0)
    pbo = float((logit_arr < 0).mean())

    return {
        'pbo': pbo,
        'mean_logit': float(np.mean(logit_arr)),
        'median_oos_rank': float(np.median(rel_ranks)),
        'pct_below_median': float((rel_ranks < 0.5).mean()),
        'n_combinations': len(combos),
        'S': S,
        'n_strategies': n_strategies,
        'logit_distribution': logit_arr.tolist(),
    }


def interpret_pbo(pbo_result: dict) -> str:
    """Translate PBO numbers into plain English."""
    pbo = pbo_result['pbo']
    if pbo < 0.2:
        verdict = "ROBUST: in-sample winners generalize well."
    elif pbo < 0.4:
        verdict = "MODERATE: some selection bias but signal is real."
    elif pbo < 0.6:
        verdict = "SUSPICIOUS: in-sample winners no better than random out-of-sample."
    else:
        verdict = "HEAVILY OVERFIT: IS-best tends to be OOS-worst."

    return (f"PBO = {pbo:.3f} (across {pbo_result['n_combinations']} splits "
            f"of {pbo_result['n_strategies']} strategies)\n  Verdict: {verdict}")
