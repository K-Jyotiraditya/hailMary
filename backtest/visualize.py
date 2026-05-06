"""
Backtest Visualization — equity curve, drawdown, monthly returns, rolling Sharpe.

Renders a single 4-panel figure that captures the strategy's full performance
profile. Saved as PNG so the user can drop it into reports.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def plot_equity_curves(curves: dict, save_path: str = None,
                       title: str = 'Equity Curves',
                       benchmark_key: str = None):
    """Single-axis equity curves, normalized to 1.0 at start."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, eq in curves.items():
        norm = eq / eq.iloc[0]
        lw = 2.0 if (benchmark_key and name == benchmark_key) else 1.6
        style = '--' if (benchmark_key and name == benchmark_key) else '-'
        ax.plot(norm.index, norm.values, label=name, linewidth=lw, linestyle=style)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Cumulative Return (1.0 = start)')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=110, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_drawdown(equity: pd.Series, save_path: str = None,
                  title: str = 'Underwater Curve (Drawdown)'):
    """Drawdown curve."""
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(dd.index, dd.values * 100, 0, color='crimson', alpha=0.5)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=110, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_monthly_returns(equity: pd.Series, save_path: str = None,
                         title: str = 'Monthly Returns Heatmap'):
    """Monthly returns heatmap (years x months)."""
    monthly = equity.resample('ME').last().pct_change().dropna()
    df = pd.DataFrame({
        'year': monthly.index.year,
        'month': monthly.index.month,
        'ret': monthly.values * 100,
    })
    pivot = df.pivot(index='year', columns='month', values='ret')
    fig, ax = plt.subplots(figsize=(11, max(3, 0.5 * len(pivot))))
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn',
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title, fontsize=14)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if pd.isna(v):
                continue
            color = 'white' if abs(v) > vmax * 0.5 else 'black'
            ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                    color=color, fontsize=8)
    plt.colorbar(im, ax=ax, label='Monthly Return (%)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=110, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_rolling_sharpe(equity: pd.Series, window: int = 252,
                        save_path: str = None,
                        title: str = 'Rolling 1-Year Sharpe Ratio'):
    """Rolling Sharpe over time."""
    rets = equity.pct_change().dropna()
    rolling_mean = rets.rolling(window).mean() * 252
    rolling_std = rets.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = (rolling_mean / rolling_std).dropna()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color='steelblue', linewidth=1.4)
    ax.axhline(0, linestyle='--', color='black', alpha=0.4)
    ax.axhline(1, linestyle=':', color='green', alpha=0.4, label='Sharpe = 1')
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Annualized Sharpe')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(loc='upper right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=110, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def make_full_report(curves: dict, primary: str, output_dir: str = 'data/backtest_results'):
    """Render all four plots for a given strategy and save to disk."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_equity_curves(curves, save_path=f'{output_dir}/equity_curves.png',
                       benchmark_key='Equal-Weight 46')
    if primary in curves:
        eq = curves[primary]
        plot_drawdown(eq, save_path=f'{output_dir}/drawdown_{primary}.png',
                      title=f'Drawdown — {primary}')
        plot_monthly_returns(eq, save_path=f'{output_dir}/monthly_returns_{primary}.png',
                              title=f'Monthly Returns — {primary}')
        plot_rolling_sharpe(eq, save_path=f'{output_dir}/rolling_sharpe_{primary}.png',
                             title=f'Rolling 1-Year Sharpe — {primary}')
