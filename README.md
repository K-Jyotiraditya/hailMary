# HailMary — Systematic Equity Alpha Engine for NIFTY 46

A full-stack ML-driven trading system for Indian large-cap equities. **Three production presets** to pick your risk/return tradeoff:

| Mode | Sharpe | CAGR | MaxDD | Calmar | When to use |
|---|---|---|---|---|---|
| **`calmar`** | 1.26 | 14.5% | **-11.1%** | **1.31** | Drawdown-conservative |
| **`balanced`** (default) | 1.33 | 17.4% | -16.5% | 1.06 | Reasonable middle |
| **`sharpe`** | **1.38** | **18.6%** | -20.6% | 0.90 | Max risk-adjusted return |

All validated with de Prado's PBO/DSR methodology over 7+ years out-of-sample.

**Production entry point:** `scripts/live_signal_final.py --mode {calmar|balanced|sharpe}`.

---

## Quick start

```bash
# 1. Install
python -m venv venv
source venv/Scripts/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Build the pipeline (one-time, ~30 min total)
PYTHONPATH=. python scripts/01_fetch_data.py         # NIFTY OHLCV
PYTHONPATH=. python scripts/02_engineer_features.py  # 65 per-stock features
PYTHONPATH=. python scripts/p2_train.py              # Cross-sectional infra
PYTHONPATH=. python scripts/p3_pipeline.py           # balanced models
PYTHONPATH=. python scripts/p4_tune.py               # Optuna hyperparameter search
PYTHONPATH=. python scripts/p5_final.py              # sharpe + calmar models

# 3. Generate today's signal — pick your risk preset
PYTHONPATH=. python scripts/live_signal_final.py --mode balanced --capital 500000
PYTHONPATH=. python scripts/live_signal_final.py --mode sharpe --capital 500000
PYTHONPATH=. python scripts/live_signal_final.py --mode calmar --capital 500000
```

---

## What the production live signal looks like

```
TARGET PORTFOLIO — 2026-04-30  Capital: Rs.500,000

Symbol         Weight Sector       Pred  Conf      Dollars   Shares
M&M            15.24% Auto         0.78  0.99    Rs.76,181       24
BPCL           14.76% Energy       0.80  0.99    Rs.73,819      245
WIPRO          10.28% IT           1.00  0.99    Rs.51,420      256
TCS             8.86% IT           0.94  0.99    Rs.44,300       17
INFY            8.26% IT           0.96  0.99    Rs.41,315       34
... (10 names total)

Sector breakdown:
    Financials    35.0%  #################
    IT            35.0%  #################
    Auto          15.2%  #######
    Energy        14.8%  #######

Why these picks (top features):
  M&M    -> drawdown_252(-2.7), vol_60(+2.0), sma_200(+1.5)
  BPCL   -> vol_60(+2.7), drawdown_252(+2.3), drawdown_60(-2.4)
```

Plus drift monitoring (warns if today's features look unlike training data) and confidence intervals (std-dev of bagged predictions).

---

## Architecture

| Layer | Module |
|---|---|
| Data | `data/{fetcher,cleaner,calendar,loader,sectors}.py` |
| Features | `features/{technical,price_based,ml_features,cross_sectional,market_context,feature_validator}.py` |
| Labels | `labels/{rank_label,triple_barrier}.py` |
| ML | `ml/{cs_model,bagging,rf_model,ensemble,cpcv}.py` |
| Portfolio | `portfolio/{optimizer,top_k,sector_neutral,regime_filter,rebalancer,vol_targeting,position_sizer}.py` |
| Backtest | `backtest/{engine,results,visualize}.py` |
| Validation | `validation/{walk_forward,pbo,dsr,robustness}.py` |
| Production | `scripts/p3_pipeline.py`, `scripts/live_signal_p3.py` |

---

## The strategy in one paragraph

Take 46 NIFTY stocks. Compute 65 per-stock features (35 technical + 27 price/return + 3 volume), then their cross-sectional ranks within the universe each day, plus 14 market-context features. Total 144 features. Train an ensemble of 5 bagged LightGBM regressors + 1 Random Forest on a panel target: the percentile rank of forward 5-day return within the universe. Retrain each year, walk-forward. At each monthly rebalance: pick the top 10 stocks by predicted rank, weight them via Hierarchical Risk Parity, cap each sector at 35%. Apply the NIFTY 50 200-DMA regime filter — if the broader market is below its 200-DMA, hold cash. Realistic 1bp slippage + 5bp commission per trade.

---

## Validation summary (out-of-sample 2019-01 → 2026-04)

- Sharpe Ratio: **1.33** (95% CI [0.56, 2.09], P(≤0) = 0.03%)
- Max Drawdown: -16.46% | Calmar: 1.06 | Sortino: 1.61
- Volatility: 12.69% (vs benchmark 17.17%)
- DSR (Deflated Sharpe Ratio): **1.0000** — gold-standard significance after multi-trial bias adjustment
- Robust to costs: Sharpe stays in [1.23, 1.36] across all reasonable slippage/commission combos
- Beats Equal-Weight 46 baseline (Sharpe 1.18 → 1.33, MaxDD -36.5% → -16.5%)

---

## What we tried that didn't work (and why it's instructive)

**Phase 4** (`scripts/p4_*.py`) added Optuna hyperparameter search, IC-based feature selection, and multi-horizon ensembling. CV showed improved IC, but the actual backtest Sharpe DROPPED from 1.33 to 1.21.

The lesson: PBO = 0.91 in Phase 2 had warned us specific parameters don't generalize. Hyperparameter optimization is overfitting on financial data, even with proper walk-forward CV. **Use sensible defaults; put effort into universe, label design, and risk overlays.**

---

## Reports (chronological)

| Report | What's in it |
|---|---|
| `REPORT.md` | Phase 1: per-stock binary classifier, ties benchmark |
| `REPORT_P2.md` | Phase 2: cross-sectional rank, top-K, regime filter — **the big jump** |
| `REPORT_P21.md` | Phase 2.1: ensemble, vol-targeting, first live signal |
| `REPORT_P3.md` | Phase 3: bagging, sector neutrality, drift monitor, viz — **production** |
| `REPORT_FINAL.md` | Unified narrative + Phase 4 over-optimization study |

---

## Key learnings

1. **Per-stock training averages out signal.** Pooled training on stacked NIFTY data gave AUC ≈ 0.50. Cross-sectional ranking jumped to mean IC 0.02-0.05 per stock and Sharpe 1.33.
2. **Walk-forward retraining is non-negotiable.** First Phase 2 backtest had Sharpe 1.70 — pure look-ahead. Honest version: 1.33.
3. **CPCV vs naive K-fold is real.** Without purging and embargoing, our cross-validation would have been inflated by overlapping label horizons.
4. **The regime filter does most of the heavy lifting.** -42% → -17% MaxDD by simply not trading when NIFTY is below its 200-DMA.
5. **Sector caps are free risk reduction.** -17.85% → -16.46% MaxDD with no Sharpe cost.
6. **Stop optimizing.** PBO told us; tuning confirmed it.

---

## What's next (not implemented)

- **Fundamental features** (P/E rank, EPS surprise) via paid feed — biggest expected lift.
- **NIFTY 100/200 universe** — more cross-sectional dispersion.
- **Live broker integration** (Zerodha Kite API) — automated execution.
- **Daily IC dashboard** — track real-time vs backtested IC for drift detection.
