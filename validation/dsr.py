"""
Deflated Sharpe Ratio (DSR) — Bailey & López de Prado (2014).

THE PROBLEM DSR SOLVES:
  A reported Sharpe is conditional on you choosing it from N tries. Even when
  the true Sharpe is 0, the maximum across N independent trials is positive
  in expectation (and grows with N). DSR strips out that selection bias to
  give a probability that the strategy's TRUE Sharpe exceeds a benchmark.

INPUTS:
  observed_sharpe: The Sharpe you measured.
  T: Sample size (number of returns).
  skewness, kurtosis: Excess kurtosis of returns (kurtosis = 0 means normal).
  n_trials: How many strategy variants you tried (be honest!).
  variance_of_trials: Variance of Sharpe ratios across the trials.
  benchmark_sharpe: What "good enough" means (usually 0).

OUTPUT:
  DSR in [0, 1] = probability that the true Sharpe exceeds benchmark.
  DSR > 0.95 is the gold standard for "this is real, not lucky."
"""
import numpy as np
from scipy import stats


def expected_max_sharpe(n_trials: int, variance_of_sharpes: float) -> float:
    """
    Expected value of max(Sharpe) across n_trials i.i.d. trials with given variance.
    Approximation via Bailey & de Prado eq. 5.

    Uses Euler-Mascheroni constant gamma ~ 0.5772 and inverse normal CDF.
    """
    if n_trials <= 1:
        return 0.0
    gamma = 0.5772156649  # Euler-Mascheroni
    # Inverse normal CDF approximation
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    expected = np.sqrt(variance_of_sharpes) * ((1 - gamma) * z1 + gamma * z2)
    return float(expected)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    T: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
    n_trials: int = 1,
    variance_of_sharpes: float = 1.0,
    benchmark_sharpe: float = 0.0,
) -> dict:
    """
    Compute Deflated Sharpe Ratio.

    The formula adjusts the Sharpe-test statistic for non-normality and
    multi-trial selection bias, then converts to a probability via the
    normal CDF.
    """
    # Expected maximum Sharpe under H0 (true Sharpe = benchmark for all trials)
    e_max = expected_max_sharpe(n_trials, variance_of_sharpes)
    deflated_benchmark = benchmark_sharpe + e_max

    # Variance of the Sharpe ratio estimator (Mertens 2002 / de Prado)
    sr_var = (1 - skewness * observed_sharpe + (excess_kurtosis / 4) * observed_sharpe**2) / (T - 1)
    if sr_var <= 0:
        return {
            'dsr': np.nan,
            'observed_sharpe': observed_sharpe,
            'deflated_benchmark': deflated_benchmark,
            'expected_max_sharpe': e_max,
            'note': 'Invalid Sharpe variance (T too small or extreme moments)'
        }

    # Z-score and probability
    z = (observed_sharpe - deflated_benchmark) / np.sqrt(sr_var)
    dsr = float(stats.norm.cdf(z))

    return {
        'dsr': dsr,
        'observed_sharpe': observed_sharpe,
        'deflated_benchmark': deflated_benchmark,
        'expected_max_sharpe': e_max,
        'z_score': z,
        'T': T,
        'n_trials': n_trials,
        'skewness': skewness,
        'excess_kurtosis': excess_kurtosis,
    }


def dsr_from_returns(returns, n_trials: int = 1, variance_of_sharpes: float = 1.0,
                     benchmark_sharpe: float = 0.0) -> dict:
    """Convenience: compute DSR directly from a return series."""
    import pandas as pd
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    returns = returns.dropna()

    if len(returns) < 30 or returns.std() == 0:
        return {'dsr': np.nan, 'note': 'Insufficient data'}

    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    skew = returns.skew()
    kurt = returns.kurtosis()  # pandas returns excess kurtosis already
    return deflated_sharpe_ratio(
        observed_sharpe=sharpe,
        T=len(returns),
        skewness=skew,
        excess_kurtosis=kurt,
        n_trials=n_trials,
        variance_of_sharpes=variance_of_sharpes,
        benchmark_sharpe=benchmark_sharpe,
    )


def interpret_dsr(dsr_result: dict) -> str:
    """Plain-English interpretation."""
    dsr = dsr_result.get('dsr', np.nan)
    if np.isnan(dsr):
        return "DSR could not be computed."
    if dsr > 0.95:
        verdict = "STRONG: very likely the true Sharpe exceeds benchmark."
    elif dsr > 0.80:
        verdict = "MODERATE: probable real edge but not gold-standard."
    elif dsr > 0.50:
        verdict = "WEAK: barely above selection-bias-adjusted noise."
    else:
        verdict = "FAIL: indistinguishable from luck after multi-trial adjustment."

    return (f"DSR = {dsr:.4f}\n  Observed Sharpe: {dsr_result['observed_sharpe']:.3f}\n"
            f"  Deflated benchmark: {dsr_result['deflated_benchmark']:.3f}\n"
            f"  Verdict: {verdict}")
