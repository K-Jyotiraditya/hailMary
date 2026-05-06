"""
Backtest Performance Metrics.

Standard institutional metrics: Sharpe, Sortino, Calmar, max drawdown,
hit rate, plus a few extras (return concentration, tail risk).

Sharpe convention: annualized using sqrt(252). Risk-free rate is implicit
in `excess_returns`; default treats raw returns as excess.
"""
import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series) -> float:
    """Geometric mean annualized return."""
    n_years = len(returns) / 252
    if n_years <= 0:
        return 0.0
    cum = (1 + returns).prod()
    return cum ** (1 / n_years) - 1 if cum > 0 else -1


def annualized_volatility(returns: pd.Series) -> float:
    """Annualized standard deviation of returns."""
    return returns.std() * np.sqrt(252)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """
    Annualized Sharpe ratio.
    risk_free is annualized; converted to daily for the excess calculation.
    """
    daily_rf = (1 + risk_free) ** (1 / 252) - 1
    excess = returns - daily_rf
    vol = excess.std()
    if vol == 0:
        return 0.0
    return (excess.mean() / vol) * np.sqrt(252)


def sortino_ratio(returns: pd.Series, target: float = 0.0) -> float:
    """
    Sortino: like Sharpe but only penalizes DOWNSIDE volatility.
    target is daily target return.
    """
    excess = returns - target
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return np.inf if excess.mean() > 0 else 0.0
    return (excess.mean() / downside.std()) * np.sqrt(252)


def max_drawdown(equity: pd.Series) -> tuple:
    """
    Maximum peak-to-trough drawdown.
    Returns (max_dd_value, peak_date, trough_date, recovery_date_or_None).
    """
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    trough = drawdown.idxmin()
    peak = equity[:trough].idxmax() if trough != equity.index[0] else trough

    # Recovery: first date after trough where equity reaches the previous peak
    recovery = None
    after_trough = equity.loc[trough:]
    peak_value = equity.loc[peak]
    recovery_idx = after_trough[after_trough >= peak_value].index
    if len(recovery_idx) > 0:
        recovery = recovery_idx[0]

    return float(drawdown.min()), peak, trough, recovery


def calmar_ratio(returns: pd.Series, equity: pd.Series) -> float:
    """Annual return / |max drawdown|. Higher is better."""
    ann_ret = annualized_return(returns)
    max_dd, *_ = max_drawdown(equity)
    if max_dd == 0:
        return np.inf
    return ann_ret / abs(max_dd)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of days with positive return."""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    """Sum of gains / |sum of losses|. > 1 means net profitable."""
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else 0.0
    return gains / losses


def value_at_risk(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical VaR at confidence level alpha (negative number)."""
    return float(returns.quantile(alpha))


def expected_shortfall(returns: pd.Series, alpha: float = 0.05) -> float:
    """Conditional VaR / Expected Shortfall (mean of worst alpha tail)."""
    var = value_at_risk(returns, alpha)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var


def compute_metrics(equity: pd.Series, label: str = '') -> dict:
    """
    Run the full metric suite on an equity curve.
    """
    returns = equity.pct_change().dropna()

    max_dd_val, peak, trough, recovery = max_drawdown(equity)

    metrics = {
        'label': label,
        'total_return': float(equity.iloc[-1] / equity.iloc[0] - 1),
        'cagr': annualized_return(returns),
        'volatility': annualized_volatility(returns),
        'sharpe': sharpe_ratio(returns),
        'sortino': sortino_ratio(returns),
        'calmar': calmar_ratio(returns, equity),
        'max_drawdown': max_dd_val,
        'max_dd_peak': peak,
        'max_dd_trough': trough,
        'max_dd_recovery': recovery,
        'hit_rate': hit_rate(returns),
        'profit_factor': profit_factor(returns),
        'var_5pct': value_at_risk(returns, 0.05),
        'es_5pct': expected_shortfall(returns, 0.05),
        'n_trading_days': len(returns),
    }
    return metrics


def print_metrics(metrics: dict):
    """Pretty-print a metrics dictionary."""
    label = metrics.get('label', 'Strategy')
    print(f"\n{'=' * 60}")
    print(f"PERFORMANCE: {label}")
    print(f"{'=' * 60}")
    print(f"  Total Return:    {metrics['total_return']:>10.2%}")
    print(f"  CAGR:            {metrics['cagr']:>10.2%}")
    print(f"  Volatility:      {metrics['volatility']:>10.2%}")
    print(f"  Sharpe Ratio:    {metrics['sharpe']:>10.3f}")
    print(f"  Sortino Ratio:   {metrics['sortino']:>10.3f}")
    print(f"  Calmar Ratio:    {metrics['calmar']:>10.3f}")
    print(f"  Max Drawdown:    {metrics['max_drawdown']:>10.2%}")
    print(f"     Peak:         {metrics['max_dd_peak']}")
    print(f"     Trough:       {metrics['max_dd_trough']}")
    if metrics['max_dd_recovery']:
        print(f"     Recovery:     {metrics['max_dd_recovery']}")
    else:
        print(f"     Recovery:     not yet recovered")
    print(f"  Hit Rate:        {metrics['hit_rate']:>10.2%}")
    print(f"  Profit Factor:   {metrics['profit_factor']:>10.3f}")
    print(f"  VaR (5%):        {metrics['var_5pct']:>10.2%}")
    print(f"  ES (5%):         {metrics['es_5pct']:>10.2%}")


def compare_strategies(strategies: dict) -> pd.DataFrame:
    """
    strategies: {name: equity_series}. Returns a DataFrame side-by-side.
    """
    rows = []
    for name, equity in strategies.items():
        m = compute_metrics(equity, label=name)
        rows.append({
            'strategy': name,
            'cagr': m['cagr'],
            'sharpe': m['sharpe'],
            'sortino': m['sortino'],
            'calmar': m['calmar'],
            'max_dd': m['max_drawdown'],
            'volatility': m['volatility'],
            'hit_rate': m['hit_rate'],
        })
    return pd.DataFrame(rows).set_index('strategy').round(4)
