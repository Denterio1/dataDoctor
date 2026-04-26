"""
feature_importance.py — SHAP-based Feature Importance.

Uses SHAP (SHapley Additive exPlanations) to explain
which features matter most for the ML model.

SHAP provides:
    - Global feature importance (which features matter overall)
    - Direction (does high value = higher or lower prediction?)
    - Magnitude (how much does each feature affect the output?)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Any

from sklearn.ensemble        import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.impute          import SimpleImputer

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def compute_feature_importance(
    data:       dict[str, Any],
    target_col: str,
    task_type:  str = "auto",
    max_rows:   int = 5_000,
) -> dict[str, Any]:
    """
    Compute SHAP feature importance for a dataset.

    Args:
        data:       Structured data dict from loader.
        target_col: Name of the target column.
        task_type:  "classification", "regression", or "auto".
        max_rows:   Maximum rows to use.

    Returns:
        {
            "features":    list of dicts sorted by importance,
            "method":      "shap" or "gini",
            "task_type":   str,
            "target":      str,
            "summary":     str,
            "top_feature": str,
        }
    """
    df = data["df"].copy()

    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # ── Encode text columns ───────────────────────────────────────────────────
    encoders = {}
    for col in X.select_dtypes(include=["object", "string"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    feature_names = list(X.columns)

    # ── Detect task type ──────────────────────────────────────────────────────
    if task_type == "auto":
        n_unique = y.nunique()
        task_type = "classification" if (y.dtype == object or n_unique <= 20) else "regression"

    # ── Encode target ─────────────────────────────────────────────────────────
    if task_type == "classification":
        le_y = LabelEncoder()
        y    = le_y.fit_transform(y.astype(str))
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        y     = pd.to_numeric(y, errors="coerce")
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    # ── Drop nulls ────────────────────────────────────────────────────────────
    mask = ~pd.isnull(y)
    X_clean = X[mask].fillna(X.median(numeric_only=True))
    y_clean = y[mask]

    if len(X_clean) < 5:
        raise ValueError("Not enough data for feature importance.")

    # ── Train model ───────────────────────────────────────────────────────────
    model.fit(X_clean, y_clean)

    # ── Compute importance ────────────────────────────────────────────────────
    method = "gini"
    importances = model.feature_importances_

    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_clean)
            shap_arr = np.array(shap_values, dtype=float)

            if shap_arr.ndim == 3:
                shap_vals = np.abs(shap_arr).mean(axis=(0, 1))
            elif shap_arr.ndim == 2:
                shap_vals = np.abs(shap_arr).mean(axis=0)
            else:
                raise ValueError("Unexpected SHAP shape")

            if shap_vals.ndim == 1 and shap_vals.shape[0] == len(feature_names):
                importances = shap_vals / shap_vals.sum() if shap_vals.sum() > 0 else importances
                method = "shap"
        except Exception:
            method = "gini"
    # ── Build feature list ────────────────────────────────────────────────────
    features = []
    for name, imp in zip(feature_names, importances):
        features.append({
            "feature":    name,
            "importance": round(float(imp), 6),
            "pct":        0.0,
        })

    features.sort(key=lambda x: x["importance"], reverse=True)

    total = sum(f["importance"] for f in features)
    for f in features:
        f["pct"] = round(f["importance"] / total * 100, 2) if total > 0 else 0.0

    top = features[0]["feature"] if features else "unknown"

    summary = (
        f"Top feature: '{top}' explains {features[0]['pct']:.1f}% of predictions. "
        f"Method: {method.upper()}."
    )

    return {
        "features":    features,
        "method":      method,
        "task_type":   task_type,
        "target":      target_col,
        "summary":     summary,
        "top_feature": top,
        "n_features":  len(features),
    }