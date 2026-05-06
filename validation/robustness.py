"""
Robustness Sweeps — vary one parameter at a time and check if Sharpe is stable.

A robust strategy's parameters live on a SMOOTH surface, not a knife-edge.
If your Sharpe collapses when you change slippage from 1bp to 2bps, you have
a fragile strategy. If it stays in a tight band across reasonable parameter
ranges, the edge is real and tradable.
"""
import pandas as pd
import numpy as np
from typing import Callable
from dataclasses import dataclass


@dataclass
class SensitivityResult:
    parameter: str
    values: list
    sharpes: list
    cagrs: list
    drawdowns: list

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            self.parameter: self.values,
            'sharpe': self.sharpes,
            'cagr': self.cagrs,
            'max_drawdown': self.drawdowns,
        })

    def stability(self) -> dict:
        """Coefficient of variation of Sharpe across the sweep."""
        sharpe_arr = np.array(self.sharpes)
        if sharpe_arr.std() == 0 or sharpe_arr.mean() == 0:
            cv = 0.0
        else:
            cv = float(sharpe_arr.std() / abs(sharpe_arr.mean()))
        return {
            'parameter': self.parameter,
            'mean_sharpe': float(sharpe_arr.mean()),
            'std_sharpe': float(sharpe_arr.std()),
            'sharpe_cv': cv,
            'min_sharpe': float(sharpe_arr.min()),
            'max_sharpe': float(sharpe_arr.max()),
            'is_stable': cv < 0.30,  # Sharpe varies < 30% across parameter range
        }


def sweep_parameter(
    backtest_runner: Callable,
    parameter: str,
    values: list,
    base_kwargs: dict,
) -> SensitivityResult:
    """
    Vary one parameter, hold others fixed, run backtest and collect Sharpe/CAGR/MaxDD.

    Args:
        backtest_runner: Callable(**kwargs) -> dict with keys 'sharpe', 'cagr', 'max_drawdown'.
        parameter: Name of parameter to vary.
        values: List of values to test for that parameter.
        base_kwargs: Other parameters held fixed.
    """
    sharpes, cagrs, drawdowns = [], [], []
    for v in values:
        kwargs = {**base_kwargs, parameter: v}
        result = backtest_runner(**kwargs)
        sharpes.append(result['sharpe'])
        cagrs.append(result['cagr'])
        drawdowns.append(result['max_drawdown'])

    return SensitivityResult(parameter=parameter, values=values,
                             sharpes=sharpes, cagrs=cagrs, drawdowns=drawdowns)


def cost_stress_test(
    backtest_runner: Callable,
    base_kwargs: dict,
    slippage_grid=(0.5, 1.0, 2.0, 5.0),
    commission_grid=(2.5, 5.0, 10.0),
) -> pd.DataFrame:
    """
    Sweep slippage X commission grid. Catches strategies that only work
    under unrealistic cost assumptions.
    """
    rows = []
    for slip in slippage_grid:
        for comm in commission_grid:
            kwargs = {**base_kwargs, 'slippage_bps': slip, 'commission_bps': comm}
            result = backtest_runner(**kwargs)
            rows.append({
                'slippage_bps': slip,
                'commission_bps': comm,
                'sharpe': result['sharpe'],
                'cagr': result['cagr'],
                'max_drawdown': result['max_drawdown'],
            })
    return pd.DataFrame(rows)
