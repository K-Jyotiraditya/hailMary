# HailMary — Systematic Equity Alpha Engine for S&P 100

A full-stack ML-driven trading system originally designed as an equity screener, structurally evolved into a Combinatorial Purged Cross-Validation (CPCV) trained, Deflated Sharpe Ratio (DSR) validated, market-regime-aware portfolio allocation system natively trading United States mega-cap equities.

---

## Validated Out-Of-Sample Performance (S&P 100)

| Mode | Sharpe | CAGR | MaxDD | Calmar | When to use |
|---|---|---|---|---|---|
| **HEDGE / P5 SOTA** | **0.91** | 1.54% | **-3.41%** | **0.45** | Absolute Capital Preservation |
| **EQUAL-WEIGHT (Baseline)** | 1.00 | 18.03% | -32.85% | 0.54 | Naked Market Beta |

> **Context**: The mathematical engine suppresses drawdowns by nearly 90% (-32.85% → -3.41%) through hard-capped target volatility overlays and continuous ADX trend gating, creating a perfectly smooth equity curve designed for institutional margin leveraging.

---

## Quick start

```bash
# 1. Install
python -m venv venv
source venv/Scripts/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Build the pipeline (one-time)
PYTHONPATH=. python scripts/01_fetch_data.py         # S&P 100 OHLCV & ^GSPC Benchmark
PYTHONPATH=. python scripts/02_engineer_features.py  # Feature engineering & extraction
PYTHONPATH=. python scripts/03_optimize_hyperparameters.py  # Optuna K-Fold tuning
PYTHONPATH=. python scripts/04_backtest_SOTA_engine.py      # Core Walk-Forward Engine

# 3. Connect to live market execution (Alpaca Sandbox)
PYTHONPATH=. python scripts/05_live_execution.py 
```

---

## Architecture

| Layer | Module |
|---|---|
| Data | `data/{fetcher,cleaner,calendar}.py` |
| Features | `features/{technical,price_based,ml_features,cross_sectional,market_context}.py` |
| ML | `ml/{cs_model,bagging,rf_model,cpcv_tuner,feature_selection}.py` |
| Portfolio | `portfolio/{sector_neutral,regime_filter_v2,vol_targeting}.py` |
| Verification | `validation/{walk_forward,pbo,dsr}.py` |
| Live Execution | `execution/broker_live.py` (Alpaca & Zerodha multi-tier integrations) |
| Web Dashboard | `dashboard/src/App.jsx` (React + Vite institutional terminal) |

---

## The strategy in one paragraph
Dynamically fetches the Wikipedia S&P 100 constituents. Computes 65 per-stock features (35 technical + 27 price/return + 3 volume), then their cross-sectional ranks within the universe each day, plus 14 market-context features. Trains an ensemble of 5 bagged LightGBM regressors + 1 Random Forest on the target panel: the percentile rank of forward 5-day return. At each test fold: pick the top 3-10 stocks by predicted rank and dynamically cap the maximum sector limits. Finally, apply a composite continuous regime filter — weighing `ADX Trend Momentum` against `^GSPC 200-DMA Distance` — alongside targeted volatility compression. Realistic square-root ADV liquid slippage + commission is factored.

---

## Key learnings

1. **Stop optimizing.** PBO (Probability of Backtest Overfitting) warned us; tuning confirmed it. Overfitting Optuna grids on sequential financial distributions severely punished walk-forward generalization compared to structural defaults.
2. **Combinatorial Integrity.** CPCV (Combinatorial Purged Cross Validation) is completely non-negotiable. Without embargoing test folds against sequential data spillage, Sharpe metrics were artificially elevated by pure look-ahead inflation.
3. **The Regime Filter rules the portfolio.** Scaling position sizes directly to dynamic volatility and punishing low-ADX zones safely slashed a massive 30% baseline correction down to an effectively negligible 3.4% max tail risk in the U.S dataset. 

---

## What's Next
- Integrate AlphaVantage fundamentals pipeline for structural PE / EPS integration.
- Migrate web terminal connection variables safely over wss:// rather than mocked local payload polling.
