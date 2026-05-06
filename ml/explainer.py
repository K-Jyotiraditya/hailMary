"""
SHAP-based feature attribution for trained LightGBM models.

SHAP values explain each prediction as the contribution of each feature
relative to the dataset mean. Aggregating |SHAP| values across many samples
gives a global importance measure that respects feature interactions, which
gain-based importance does not.
"""
import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def shap_importance(model, X: pd.DataFrame, max_samples: int = 2000) -> pd.DataFrame:
    """
    Compute mean |SHAP| value per feature on a sample of X.

    Returns DataFrame with columns: feature, mean_abs_shap, sign_consistency
    """
    if not HAS_SHAP:
        raise ImportError("Install shap: pip install shap")

    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # LightGBM binary: shap_values may be list of two arrays or single array
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values
        if sv.ndim == 3:
            sv = sv[:, :, -1]  # positive class

    abs_shap = np.abs(sv).mean(axis=0)
    sign_consistency = np.abs((sv > 0).mean(axis=0) - 0.5) * 2  # 0=mixed, 1=consistent

    return pd.DataFrame({
        'feature': X.columns,
        'mean_abs_shap': abs_shap,
        'sign_consistency': sign_consistency,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
