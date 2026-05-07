"""
XGBoost Direction Model Trainer.

Walk-forward train/test split (never leaks future data into training).
Trains on first 80% of dates, tests on last 20%.
Saves model + scaler to models/saved/.

Run:  python models/xgboost_trainer.py
      python models/xgboost_trainer.py --period 2y   (more data)
"""
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data.feature_engineer import build_features, FEATURE_COLS, FORWARD_DAYS

try:
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("Run: python -m pip install xgboost scikit-learn")
    sys.exit(1)

SAVE_DIR = Path(__file__).parent / "saved"
MODEL_PATH  = SAVE_DIR / "xgb_direction.pkl"
SCALER_PATH = SAVE_DIR / "scaler.pkl"
FEATURES_PATH = SAVE_DIR / "feature_cols.pkl"


def train(period: str = "1y", test_frac: float = 0.20):
    print("\n" + "=" * 60)
    print("  XGBoost Direction Model — Training Pipeline")
    print("=" * 60 + "\n")

    # ── 1. Build feature matrix ───────────────────────────────────────────────
    df = build_features(period=period)
    df = df.sort_values("date").reset_index(drop=True)

    n_total = len(df)
    n_train = int(n_total * (1 - test_frac))
    train_df = df.iloc[:n_train].copy()
    test_df  = df.iloc[n_train:].copy()

    print(f"  Period:    {df['date'].min().date()}  to  {df['date'].max().date()}")
    print(f"  Stocks:    {df['ticker'].nunique()}")
    print(f"  Samples:   {n_total} total | {n_train} train | {n_total - n_train} test")
    print(f"  Pos rate:  {df['label'].mean():.1%} (days stock beats SPY in 5d)\n")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label"].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df["label"].values

    # ── 2. Scale ──────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── 3. Train ──────────────────────────────────────────────────────────────
    print("  Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_weight=15,    # prevents overfit on small groups
        reg_lambda=2.0,
        reg_alpha=0.5,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=False,
    )

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    preds = model.predict(X_test_s)
    proba = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    test_df = test_df.copy()
    test_df["proba"] = proba

    # Precision@Top-5: each day, pick top-5 by proba, what fraction went up?
    p5_list = []
    for _, grp in test_df.groupby("date"):
        if len(grp) < 2:
            continue
        top = grp.nlargest(min(5, len(grp)), "proba")
        p5_list.append(top["label"].mean())
    p_at_5 = float(np.mean(p5_list)) if p5_list else 0.0

    # Information Coefficient (rank correlation between proba and fwd_ret)
    try:
        from scipy.stats import spearmanr
        valid = test_df["fwd_ret"].notna()
        ic, _ = spearmanr(proba[valid], test_df.loc[valid, "fwd_ret"].values)
        ic = float(ic) if np.isfinite(ic) else 0.0
    except Exception:
        ic = 0.0

    print(f"\n  {'Metric':<25} {'Value':>10}")
    print(f"  {'-'*35}")
    print(f"  {'Accuracy':<25} {acc:>10.3f}")
    print(f"  {'ROC-AUC':<25} {auc:>10.3f}")
    print(f"  {'Precision@Top-5':<25} {p_at_5:>10.3f}")
    print(f"  {'Info Coefficient (IC)':<25} {ic:>10.4f}")

    # ── 5. Backtest ───────────────────────────────────────────────────────────
    _backtest(test_df)

    # ── 6. Feature importance ─────────────────────────────────────────────────
    print("\n  Feature importances:")
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    for feat, val in imp.items():
        bar = "#" * int(val * 200)
        print(f"    {feat:<25} {val:.4f}  {bar}")

    # ── 7. Quality gate ───────────────────────────────────────────────────────
    # IC < 0 means predictions are inversely correlated with returns — model
    # likely found regime-specific patterns that reversed in the test set.
    # We save it but flag it; pipeline will fall back to LLM if quality is poor.
    model_is_usable = (auc >= 0.50) and (ic >= 0.0)
    quality = {"auc": round(auc, 4), "ic": round(ic, 4),
               "acc": round(acc, 4), "usable": model_is_usable}
    if not model_is_usable:
        print(f"\n  [!] Model quality below threshold (AUC={auc:.3f}, IC={ic:.4f}).")
        print("      Pipeline will fall back to LLM until model improves.")
        print("      Tip: relative return label + insider_score usually flips IC positive.\n")

    # ── 8. Save ───────────────────────────────────────────────────────────────
    SAVE_DIR.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(FEATURE_COLS, f)
    with open(SAVE_DIR / "quality.pkl", "wb") as f:
        pickle.dump(quality, f)

    print(f"  Saved to models/saved/")
    return model, scaler


def _backtest(test_df: pd.DataFrame):
    """Simulate daily top-5 equal-weight portfolio vs SPY on the test set."""
    import yfinance as yf

    dates = sorted(test_df["date"].unique())
    if len(dates) < 5:
        return

    start = str(pd.Timestamp(dates[0]).date())
    end   = str(pd.Timestamp(dates[-1]).date())
    raw_spy = yf.download("SPY", start=start, end=end,
                          auto_adjust=True, progress=False)
    spy_close = raw_spy["Close"]
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]
    spy_total = float(spy_close.iloc[-1] / spy_close.iloc[0] - 1) if len(spy_close) > 1 else 0.0

    # Strategy: each day take top-5 by proba, hold for FORWARD_DAYS, equal weight
    daily_rets = []
    for date in dates:
        grp = test_df[test_df["date"] == date].dropna(subset=["fwd_ret"])
        if grp.empty:
            continue
        top = grp.nlargest(min(5, len(grp)), "proba")
        daily_rets.append(float(top["fwd_ret"].mean()))

    strategy_total = float(np.prod([1 + r / FORWARD_DAYS for r in daily_rets]) - 1)

    print(f"\n  Backtest (top-5 daily, {len(dates)} days):")
    print(f"    Strategy return : {strategy_total:>+8.2%}")
    print(f"    SPY return      : {spy_total:>+8.2%}")
    print(f"    Alpha           : {strategy_total - spy_total:>+8.2%}")


def load_model():
    """Load saved model + scaler. Returns (model, scaler) or (None, None)."""
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def is_model_usable() -> bool:
    """Return True only if the saved model passed the quality threshold."""
    q_path = SAVE_DIR / "quality.pkl"
    if not q_path.exists():
        return False
    with open(q_path, "rb") as f:
        q = pickle.load(f)
    return bool(q.get("usable", False))


def predict(features: dict) -> tuple[float, str]:
    """
    Run inference on a single stock's feature dict.
    Returns (probability_up, direction) or (0.5, 'sideways') if model
    is not trained or failed the quality gate.
    """
    if not is_model_usable():
        return 0.5, "sideways"

    model, scaler = load_model()
    if model is None:
        return 0.5, "sideways"

    x = np.array([[features.get(c, 0.0) for c in FEATURE_COLS]])
    x_s = scaler.transform(x)
    prob = float(model.predict_proba(x_s)[0, 1])
    direction = "bullish" if prob >= 0.55 else ("bearish" if prob <= 0.45 else "sideways")
    return prob, direction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="10y",
                        help="yfinance period string: 2y, 5y, 10y, max (default: 10y)")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward CV first, then train final model")
    args = parser.parse_args()

    if args.walk_forward:
        from models.walk_forward import run_walk_forward
        run_walk_forward(period=args.period)

    train(period=args.period)
