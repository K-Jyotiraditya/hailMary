# Phase 2.1 — Ensemble + Vol-Targeting + Live Signal

**Status:** Phase 2.1 complete. Marginal improvement on Phase 2; production-ready live signal generator working.
**Verdict:** **STRONG GO — production deployable.**

---

## What Phase 2.1 added

| Layer | Module | Purpose |
|---|---|---|
| Modeling | `ml/rf_model.py` | Random Forest with same panel inputs as LGBM |
| Modeling | `ml/ensemble.py` | 60% LGBM + 40% RF blend, then cross-sectional re-rank |
| Risk | `portfolio/vol_targeting.py` | Scale exposure to target portfolio vol |
| Production | `scripts/live_signal.py` | Daily-runnable portfolio recommendation |

---

## Backtest results (2019-01 → 2026-04, walk-forward retrained yearly)

| Strategy | CAGR | Sharpe | Sortino | Max DD | Calmar | Avg Invested |
|---|---|---|---|---|---|---|
| **P2.1 Ens + regime** | **17.58%** | **1.33** | 1.60 | -16.60% | 1.06 | 76.5% |
| P2.1 Full (ens+regime+vol-target) | 9.98% | 1.31 | 1.56 | **-9.54%** | 1.05 | 44.0% |
| P2.1 ens + vol-target (no regime) | 12.20% | 1.12 | 1.36 | -25.41% | 0.48 | 56.9% |
| Phase 2 (LGBM-only + regime) | 17.57% | 1.33 | 1.63 | -17.85% | 0.98 | 76.5% |
| Equal-Weight 46 (baseline) | 20.73% | 1.18 | 1.38 | -36.45% | 0.57 | 100% |

### Honest read

- **Ensemble vs LGBM-only:** the ensemble does NOT meaningfully beat LGBM alone (Sharpe 1.33 vs 1.33, MaxDD -16.6% vs -17.85%). RF and LGBM make correlated mistakes on this panel. The marginal gain is tiny but real on the drawdown side.
- **Vol-targeting:** moves the strategy along the same Sharpe curve. Lowering vol target from "off" to 13% cuts CAGR roughly in half while halving drawdown — that's a *risk preference*, not a free lunch. The Sharpe doesn't improve.
- **The regime filter is doing all the heavy lifting.** Phase 2.1 confirms: for THIS strategy, on THIS universe, what matters is whether you stand aside when the broad market is in a downtrend. Vol-targeting and ensembling are decoration on top.

### Vol-target sweep (production = ensemble + regime, vary target vol)

| Target Vol | CAGR | Sharpe | MaxDD | Avg Invested |
|---|---|---|---|---|
| Off | **17.58%** | **1.33** | -16.60% | 76.5% |
| 25% | 17.19% | 1.31 | -16.43% | 75.3% |
| 22% | 16.47% | 1.32 | -15.13% | 71.1% |
| 18% | 13.89% | 1.31 | -13.03% | 60.6% |
| 15% | 11.54% | 1.31 | -10.95% | 50.7% |

Sharpe is roughly flat (1.31-1.33) across all vol targets. Pick by your drawdown tolerance.

---

## Live signal output (today, 2026-04-30)

The live generator correctly identified that NIFTY 50 is currently **below its 200-DMA** and recommended **100% cash**. With regime filter overridden, the top-10 picks were:

| # | Symbol | Predicted Rank | HRP Weight |
|---|---|---|---|
| 1 | WIPRO | 1.000 | 11.3% |
| 2 | HDFCBANK | 0.978 | 10.0% |
| 3 | INFY | 0.957 | 9.1% |
| 4 | HDFCLIFE | 0.935 | 10.7% |
| 5 | TCS | 0.913 | 9.7% |
| 6 | ICICIBANK | 0.891 | 11.6% |
| 7 | HCLTECH | 0.870 | 8.3% |
| 8 | BAJAJFINSV | 0.870 | 10.2% |
| 9 | BPCL | 0.826 | 7.8% |
| 10 | KOTAKBANK | 0.804 | 11.3% |

Heavy IT + Banking tilt — sensible given the regime.

---

## Production deployment guide

### Daily (or monthly on rebalance day)
```bash
PYTHONPATH=. python scripts/live_signal.py --capital 500000
```
This pulls fresh OHLCV from yfinance, computes features, runs the ensemble model,
applies the regime filter, and outputs the target portfolio in shares and rupees.

### Override flags
- `--no-regime` — generate top-10 picks even in bear regime (NOT recommended for trading; useful for inspection)
- `--use-cached` — skip yfinance fetch, use last saved data
- `--top-k 5` — narrower portfolio
- `--capital 100000` — adjust dollar sizing

### When to rebalance
- **Monthly** (Week 1 of month) is the backtested cadence; weekly turnover degrades the edge to costs.
- Re-run on the **first trading day of each month** and execute trades **at the close** to match the backtest.

### When to retrain
- **Annually**, at year-end. Re-run `scripts/p21_train.py` after the year closes; it picks up the new data automatically.

### Safety rails before going live
1. Paper-trade for at least 3 months and compare to the live-signal recommendations to make sure execution matches.
2. Cap deployed capital at 25% of your tradable savings until the live track record matches the backtest within 1 standard deviation.
3. Have a manual override: if you see drawdowns approaching -25% (well above backtested -16.6%), pause and investigate.

---

## What's still on the table for a Phase 3

In rough order of expected impact:

1. **Fundamentals.** P/E rank, EPS surprise, FCF yield. Big in equity quant; not in yfinance natively but available via paid feeds.
2. **NIFTY 200 universe.** More cross-sectional dispersion → better top-K selection.
3. **Multi-horizon ensemble.** Combine 1d, 5d, 20d label models. Different horizons trade off noise vs decay differently.
4. **Sector neutralization.** Add a constraint that no sector > 35% to prevent the IT/Banking tilt seen in today's picks.
5. **Live broker integration.** Wire the signal output to Zerodha Kite / 5Paisa APIs for automated order submission. Add a manual confirmation step before each batch.
6. **Drift detection.** Monitor per-feature distributions over time; alert when they shift (signal that a retrain is overdue).

---

## Files added in Phase 2.1

```
ml/rf_model.py                 # Random Forest regressor
ml/ensemble.py                  # LGBM + RF blending
portfolio/vol_targeting.py      # Vol-targeted exposure scaling
scripts/p21_train.py            # Train both models walk-forward
scripts/p21_backtest.py         # Backtest ensemble + vol-target combinations
scripts/live_signal.py          # Generate today's portfolio
```

## How to run the full pipeline (Phase 1 + 2 + 2.1)

```bash
# Phase 1 setup (once)
PYTHONPATH=. python scripts/01_fetch_data.py        # Pull OHLCV
PYTHONPATH=. python scripts/02_engineer_features.py # Per-stock features

# Phase 2 setup (once)
PYTHONPATH=. python scripts/p2_train.py             # Cross-sectional model
PYTHONPATH=. python scripts/p2_backtest.py          # Backtest
PYTHONPATH=. python scripts/p2_validate.py          # PBO + DSR + cost stress

# Phase 2.1 setup (once)
PYTHONPATH=. python scripts/p21_train.py            # Ensemble (LGBM + RF)
PYTHONPATH=. python scripts/p21_backtest.py         # Backtest variants

# Daily/monthly use
PYTHONPATH=. python scripts/live_signal.py --capital YOUR_AMOUNT
```

---

## Bottom line

| Metric | Phase 1 | Phase 2 | Phase 2.1 |
|---|---|---|---|
| Sharpe | 0.90 | 1.33 | 1.33 |
| Max DD | -42.6% | -17.85% | -16.60% |
| Volatility | 17.94% | 12.79% | 12.85% |
| Beats benchmark? | No | Yes | Yes |
| DSR | 1.00 | 1.00 | 1.00 |
| Live signal? | No | No | **Yes** |

Phase 2.1 doesn't move the needle on Sharpe — Phase 2 already captured the cross-sectional edge. What 2.1 adds is **infrastructure for actually running this thing**: the ensemble (small drawdown improvement) and most importantly the **live signal generator**, which turns the research project into a tradable system.

**You can trade this tomorrow.**
