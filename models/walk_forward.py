"""
Walk-Forward Cross-Validation for the XGBoost direction model.

Standard ML train/test splits are deceptive for financial data because
a single split captures one market regime. Walk-forward CV tests whether
the model generalizes ACROSS regimes by training on past, testing on future,
advancing the window each year.

Method: expanding window (train grows each split, test = next 1 year)

  Split 1: train 2016-2018 | test 2018-2019
  Split 2: train 2016-2019 | test 2019-2020
  ...
  Split N: train 2016-2024 | test 2024-2025

If average IC across splits is > 0.0, the model is genuinely predictive.
If IC is positive in some splits and negative in others, it's regime-dependent.

Run:  python models/walk_forward.py
      python models/walk_forward.py --period 10y
"""
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from data.feature_engineer import build_features, FEATURE_COLS

try:
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr
except ImportError:
    print("Run: python -m pip install xgboost scikit-learn scipy")
    sys.exit(1)

MIN_TRAIN_YEARS = 2
TEST_YEARS      = 1


def _train_and_eval(train_df: pd.DataFrame, test_df: pd.DataFrame,
                    split_label: str) -> dict:
    """Train XGBoost on train_df, evaluate on test_df. Returns metrics dict."""
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label"].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df["label"].values

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"split": split_label, "auc": 0.5, "ic": 0.0, "n_test": len(test_df)}

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_weight=15,
        reg_lambda=2.0,
        reg_alpha=0.5,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)

    proba = model.predict_proba(X_test_s)[:, 1]
    auc = float(roc_auc_score(y_test, proba))

    # IC: rank correlation between predicted proba and actual forward return
    ic = 0.0
    if "fwd_ret" in test_df.columns:
        valid = test_df["fwd_ret"].notna()
        if valid.sum() > 10:
            ic_val, _ = spearmanr(proba[valid.values], test_df.loc[valid, "fwd_ret"].values)
            ic = float(ic_val) if np.isfinite(ic_val) else 0.0

    return {
        "split":   split_label,
        "auc":     round(auc, 4),
        "ic":      round(ic, 4),
        "n_train": len(train_df),
        "n_test":  len(test_df),
    }


def run_walk_forward(period: str = "10y") -> list:
    """
    Build full feature matrix then run expanding-window walk-forward CV.
    Returns list of per-split metric dicts.
    """
    print("\n" + "=" * 65)
    print("  Walk-Forward Cross-Validation")
    print("=" * 65 + "\n")

    print(f"  Building features ({period} of data)...")
    df = build_features(period=period)
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    min_date = df["date"].min()
    max_date = df["date"].max()
    total_years = (max_date - min_date).days / 365.25

    print(f"  Date range: {min_date.date()} to {max_date.date()} "
          f"({total_years:.1f} years)")
    print(f"  Total samples: {len(df):,} across {df['ticker'].nunique()} stocks")
    print(f"  Pos rate: {df['label'].mean():.1%} (beats SPY)\n")

    if total_years < MIN_TRAIN_YEARS + TEST_YEARS:
        print(f"  [!] Not enough data for walk-forward (need {MIN_TRAIN_YEARS + TEST_YEARS}+ years)")
        return []

    # Build split boundaries (expanding window, 1-year test increments)
    splits = []
    test_start = min_date + pd.DateOffset(years=MIN_TRAIN_YEARS)
    while test_start + pd.DateOffset(years=TEST_YEARS) <= max_date:
        test_end = test_start + pd.DateOffset(years=TEST_YEARS)
        train_df = df[df["date"] < test_start].copy()
        test_df  = df[(df["date"] >= test_start) & (df["date"] < test_end)].copy()

        if len(train_df) >= 200 and len(test_df) >= 50:
            splits.append((train_df, test_df, test_start, test_end))

        test_start = test_end

    if not splits:
        print("  [!] No valid splits generated.")
        return []

    print(f"  Running {len(splits)} walk-forward splits...\n")
    print(f"  {'Split':<20} {'Train':>8} {'Test':>8} {'AUC':>8} {'IC':>10}")
    print(f"  {'-'*55}")

    results = []
    for train_df, test_df, ts, te in splits:
        label = f"{ts.year}-{te.year}"
        metrics = _train_and_eval(train_df, test_df, label)
        results.append(metrics)
        ic_flag = "+" if metrics["ic"] > 0 else "-"
        print(f"  {label:<20} {metrics['n_train']:>8,} {metrics['n_test']:>8,} "
              f"{metrics['auc']:>8.4f} {ic_flag}{abs(metrics['ic']):>9.4f}")

    # Summary statistics
    aucs = [r["auc"] for r in results]
    ics  = [r["ic"]  for r in results]
    pos_ic_splits = sum(1 for ic in ics if ic > 0)

    print(f"\n  {'Summary':<20} {'':>8} {'':>8} {'':>8} {'':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Mean':<20} {'':>8} {'':>8} {np.mean(aucs):>8.4f} {np.mean(ics):>+10.4f}")
    print(f"  {'Std':<20} {'':>8} {'':>8} {np.std(aucs):>8.4f} {np.std(ics):>10.4f}")
    print(f"  {'Min':<20} {'':>8} {'':>8} {np.min(aucs):>8.4f} {np.min(ics):>+10.4f}")
    print(f"  {'Max':<20} {'':>8} {'':>8} {np.max(aucs):>8.4f} {np.max(ics):>+10.4f}")
    print(f"\n  Positive IC splits: {pos_ic_splits}/{len(splits)}")

    mean_ic  = float(np.mean(ics))
    mean_auc = float(np.mean(aucs))
    if mean_auc >= 0.51 and mean_ic > 0.0:
        print(f"\n  [PASS] Model generalizes across regimes. "
              f"Mean AUC={mean_auc:.4f}, Mean IC={mean_ic:.4f}")
        print("         Recommend training final model on full dataset.")
    elif mean_ic > 0.0:
        print(f"\n  [MARGINAL] IC positive but AUC weak. "
              f"Some predictive signal, edge is thin.")
    else:
        print(f"\n  [FAIL] Mean IC={mean_ic:.4f} — model does not generalize. "
              f"Need stronger features (e.g. FinBERT sentiment history).")

    return results


def train_final_model(period: str = "10y"):
    """
    After walk-forward validation, train the production model on all available data.
    Saves to models/saved/ (same paths as xgboost_trainer.py).
    """
    from data.feature_engineer import FEATURE_COLS, FORWARD_DAYS

    SAVE_DIR = Path(__file__).parent / "saved"
    SAVE_DIR.mkdir(exist_ok=True)

    print(f"\n  Training final model on full {period} dataset...")
    df = build_features(period=period)
    df = df.sort_values("date").reset_index(drop=True)

    X = df[FEATURE_COLS].values
    y = df["label"].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_weight=15,
        reg_lambda=2.0,
        reg_alpha=0.5,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_s, y, verbose=False)

    with open(SAVE_DIR / "xgb_direction.pkl",  "wb") as f: pickle.dump(model,  f)
    with open(SAVE_DIR / "scaler.pkl",          "wb") as f: pickle.dump(scaler, f)
    with open(SAVE_DIR / "feature_cols.pkl",    "wb") as f: pickle.dump(FEATURE_COLS, f)

    print(f"  Final model saved to models/saved/")
    return model, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="10y",
                        help="yfinance period string (default: 10y)")
    parser.add_argument("--train-final", action="store_true",
                        help="After CV, train and save final model on full data")
    args = parser.parse_args()

    results = run_walk_forward(period=args.period)

    if args.train_final and results:
        mean_ic = np.mean([r["ic"] for r in results])
        if mean_ic > 0:
            train_final_model(period=args.period)
        else:
            print("\n  Skipping final model save — mean IC is negative.")
