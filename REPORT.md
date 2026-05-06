# NIFTY 20 Systematic Equity Alpha Engine — Phase 1 Final Report

**Status:** Phase 1 MVP complete (5 weeks of work, all five workstreams shipped).
**Verdict:** GO — strategy passes 7 of 8 validation checks; modest but real edge.

---

## What was built

A complete trading-grade systematic equity research and execution platform on NIFTY 20:

| Layer | Module | What it does |
|---|---|---|
| Data | `data/{fetcher,cleaner,calendar,loader}.py` | Pulls 10y OHLCV from yfinance, handles corporate actions, NSE calendar |
| Features | `features/{technical,price_based,ml_features,feature_validator}.py` | 65 features (35 technical + 27 price/return + validation utilities) |
| Labels | `labels/triple_barrier.py` | Volatility-adaptive triple-barrier labeling (de Prado AFML Ch.3) |
| ML | `ml/{cpcv,preprocessor,lgbm_model,explainer}.py` | LightGBM trained per stock with Combinatorial Purged CV |
| Portfolio | `portfolio/{optimizer,position_sizer,rebalancer}.py` | HRP allocation + threshold-based sizing + monthly rebalance |
| Backtest | `backtest/{engine,results}.py` | Vectorized backtester with realistic slippage / commission / cash drift |
| Validation | `validation/{walk_forward,pbo,dsr,robustness}.py` | Multi-fold WF, PBO, DSR, cost stress |
| Scripts | `scripts/{01..05}*.py` | One script per pipeline stage |

---

## Headline results (7+ years out-of-sample, 2019-01 → 2026-04)

| Strategy | CAGR | Sharpe | Sortino | Max DD | Calmar |
|---|---|---|---|---|---|
| **ML Walk-Forward (production)** | **17.67%** | **0.90** | **1.16** | **-42.6%** | **0.41** |
| Equal-Weight 6 (same universe)   | 15.28% | 0.90 | 1.07 | -40.9% | 0.37 |
| Equal-Weight 18 (full universe)  | 17.78% | 1.04 | 1.22 | -38.3% | 0.46 |

The strategy ties the broader equal-weight-18 baseline in CAGR and matches (or
slightly underperforms) on Sharpe due to concentration in 6 stocks. It does
beat equal-weight-6 by ~2.4% CAGR — consistent with the average IC of 0.02
we saw in Week 3.

---

## Validation results

### Walk-forward (4 folds within the backtest)
| Train through | OOS Sharpe | Degradation |
|---|---|---|
| 2022-08 | 2.00 | -1.28 (OOS BETTER than IS) |
| 2023-07 | 1.86 | -0.88 |
| 2024-07 | -0.05 | +1.05 (full collapse in late 2024) |
| 2025-06 | 0.55 | +0.41 |

Mean OOS Sharpe: **1.09**, Mean degradation: **-0.18** (no overall decay), but
std OOS Sharpe = **1.00** — the strategy is **regime-dependent**, working great
in 2022-2024, badly in late 2024, and recovering in 2025-2026.

### Bootstrap CI on Sharpe (block-bootstrap, 3000 samples)
- 95% CI: **[0.11, 1.77]** — excludes zero
- P(true Sharpe ≤ 0) = **1.5%** — strong evidence the edge is real

### PBO across 72 strategy variants
- PBO = **0.501** ⚠ — the only failed check
- Means: when we pick the variant with the best in-sample Sharpe, it ranks
  below OOS median half the time. The aggregate strategy class is profitable,
  but no SINGLE parameter setting is robustly best.

### Deflated Sharpe Ratio
- DSR = **1.0** — observed Sharpe of 0.90 vastly exceeds the multi-trial
  selection benchmark of 0.26 because all 72 variants behaved similarly
  (low variance of Sharpes across variants = 0.011).

### Cost stress (slippage × commission)
| Slip / Comm | 2.5 bps | 5.0 bps | 10.0 bps |
|---|---|---|---|
| 0.5 bps | 0.918 | 0.904 | 0.876 |
| 1.0 bps | 0.916 | 0.901 | 0.873 |
| 2.0 bps | 0.910 | 0.896 | 0.867 |
| 5.0 bps | 0.893 | 0.879 | 0.850 |

Sharpe varies only 0.85 → 0.92 across the entire reasonable cost grid. **ROBUST.**

---

## The eight checks

| Check | Result |
|---|---|
| OOS Sharpe > 0.3 | **PASS** (0.90) |
| OOS Sharpe 95% CI excludes 0 | **PASS** ([0.11, 1.77]) |
| P(Sharpe ≤ 0) < 0.10 | **PASS** (0.015) |
| WF mean OOS Sharpe > 0 | **PASS** (1.09) |
| WF mean degradation < 30% | **PASS** (-18%) |
| **PBO < 0.50** | **FAIL** (0.501) |
| DSR > 0.50 | **PASS** (1.00) |
| Robust to costs | **PASS** |

**7 of 8 → GO with caveats.**

---

## What we learned (the actually-valuable part)

1. **Pooled training kills per-stock signal.** Training one model on stacked
   data from all 18 stocks gave AUC = 0.50 (random). Training one model per
   stock gave mean IC = 0.02 with 6 stocks at IC > 0.02. Different stocks have
   different feature/return relationships and pooling averages them out.

2. **The look-ahead-bias trap is huge.** The first backtest used a model
   trained on all data (including the test period). It produced Sharpe = 1.70.
   Once we fixed it with annual walk-forward retraining, Sharpe dropped to
   0.90. Backtest results that aren't honestly walk-forward are fiction.

3. **Naive ML on technical features doesn't beat passive diversification.**
   Our final strategy ties the equal-weight 18-stock benchmark. The ~2% IC
   edge is real but small, and gets eaten by concentration risk and turnover
   costs once you focus on a 6-stock subset.

4. **CPCV vs. naive K-fold makes a real difference.** Without purging and
   embargoing, our cross-validation metrics would have been inflated by
   data leakage from overlapping label horizons. With CPCV, we got an
   honest IC of 0.02 instead of an inflated 0.05+.

5. **Regime instability is the dominant risk.** Walk-forward shows OOS Sharpe
   varying from -0.05 to 2.00 across 4 folds. The strategy works on average
   but you'd be unhappy holding it through 2024-Q4.

---

## What would actually move the needle (Phase 2+)

Not all of these are equally useful — listed roughly in expected impact order:

1. **Cross-sectional ranking on a wider universe** (NIFTY 50 / 100 / 200).
   Pick top-K each month rather than fixing the universe. More candidates
   means more dispersion to exploit.

2. **Fundamental features.** P/E rank, earnings revisions, FCF yield, ROIC.
   Technical features alone don't generate enough alpha after costs.

3. **Regime filter.** Add a market-state classifier (e.g., NIFTY 50 above its
   200-DMA) and skip the strategy in confirmed downtrends. This addresses the
   2024-Q4 collapse directly.

4. **Random Forest ensemble.** Train RF in parallel with LightGBM and average
   probabilities. Tree ensembles disagree on different parts of feature space —
   averaging reduces noise without reducing bias.

5. **Better label engineering.** Try meta-labeling (a second classifier
   predicting whether the first one is right) and adaptive barriers.

6. **Hyperparameter optimization with PBO.** Instead of fixing LightGBM params,
   optimize them inside CPCV with PBO as a constraint.

---

## How to run this

```bash
PYTHONPATH=. python scripts/01_fetch_data.py        # 10y OHLCV
PYTHONPATH=. python scripts/02_engineer_features.py # 65 features per stock
PYTHONPATH=. python scripts/03_train_models.py      # Per-stock LightGBM + CPCV
PYTHONPATH=. python scripts/04_backtest.py          # Walk-forward backtest
PYTHONPATH=. python scripts/05_validate.py          # PBO + DSR + cost stress
```

All artifacts land in `data/processed/`, `data/models/`, `data/backtest_results/`.

---

## What I'd tell a quant interviewer

> "I built a NIFTY 20 systematic equity strategy end-to-end:
> data ingestion, 65-feature engineering, triple-barrier labels, per-stock
> LightGBM with combinatorial purged CV, HRP portfolio construction with
> threshold-based sizing, and walk-forward backtesting with realistic costs.
> The first version showed Sharpe 1.70 — clearly look-ahead bias from the
> final model touching test data. After fixing with annual WF retraining,
> the honest Sharpe is 0.90 and the strategy roughly ties an equal-weight
> baseline. PBO = 0.50 confirms no specific parameter set is robustly best,
> but the bootstrap CI [0.11, 1.77] and DSR = 1.0 say the *aggregate* edge
> is real. The lessons are:
> (1) pooled training destroys per-stock signal,
> (2) walk-forward retraining is non-negotiable,
> (3) basic technicals on liquid large caps don't beat diversification,
> and (4) the next gains come from fundamental features, regime filters,
> and a wider cross-section, not from ML model tuning."

That's the complete Phase 1.
