"""
auto_ml.py — Advanced Auto ML Baseline.

Automatically trains and compares multiple ML models,
selects the best one, and provides detailed performance metrics.

Models tested:
    Classification: LogisticRegression, RandomForest, GradientBoosting, SVM, KNN
    Regression:     Ridge, RandomForest, GradientBoosting, SVR, KNN

Uses 5-fold StratifiedKFold cross-validation for robust evaluation.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Any

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.impute          import SimpleImputer

from sklearn.linear_model    import LogisticRegression, Ridge
from sklearn.ensemble        import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble        import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm             import SVC, SVR
from sklearn.neighbors       import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics         import classification_report


CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM":                 SVC(kernel="rbf", random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
}

REGRESSION_MODELS = {
    "Ridge Regression":  Ridge(alpha=1.0),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR":               SVR(kernel="rbf"),
    "KNN":               KNeighborsRegressor(n_neighbors=5),
}


def run_auto_ml(
    data:        dict[str, Any],
    target_col:  str,
    task_type:   str = "auto",
    max_rows:    int = 10_000,
) -> dict[str, Any]:
    """
    Train and compare multiple ML models automatically.

    Args:
        data:       Structured data dict from loader.
        target_col: Name of the target column.
        task_type:  "classification", "regression", or "auto".
        max_rows:   Maximum rows to use (samples for large datasets).

    Returns:
        {
            "task_type":    str,
            "target":       str,
            "best_model":   str,
            "best_score":   float,
            "best_metric":  str,
            "results":      list of model result dicts,
            "summary":      str,
            "recommendation": str,
        }
    """
    df = data["df"].copy()

    # ── Sample if too large ───────────────────────────────────────────────────
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)

    # ── Prepare X and y ───────────────────────────────────────────────────────
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # ── Encode text columns in X ──────────────────────────────────────────────
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # ── Detect task type ──────────────────────────────────────────────────────
    if task_type == "auto":
        n_unique = y.nunique()
        if y.dtype == object or n_unique <= 20:
            task_type = "classification"
        else:
            task_type = "regression"

    # ── Encode target for classification ──────────────────────────────────────
    if task_type == "classification":
        le_y = LabelEncoder()
        y    = le_y.fit_transform(y.astype(str))
        metric = "accuracy"
        models = CLASSIFICATION_MODELS
        cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        y      = pd.to_numeric(y, errors="coerce")
        metric = "r2"
        models = REGRESSION_MODELS
        cv     = KFold(n_splits=5, shuffle=True, random_state=42)

    # ── Drop rows where y is null ─────────────────────────────────────────────
    mask = ~pd.isnull(y)
    X    = X[mask]
    y    = y[mask] if task_type == "regression" else y[mask]

    if len(X) < 10:
        raise ValueError("Not enough data to train models (need at least 10 rows).")

    # ── Train and evaluate each model ─────────────────────────────────────────
    results: list[dict] = []

    for name, model in models.items():
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler",  StandardScaler()),
            ("model",   model),
        ])

        try:
            scores = cross_val_score(
                pipeline, X, y,
                cv=cv,
                scoring=metric,
                n_jobs=-1,
            )
            mean_score = round(float(scores.mean()), 4)
            std_score  = round(float(scores.std()), 4)

            results.append({
                "model":      name,
                "score":      mean_score,
                "std":        std_score,
                "metric":     metric,
                "cv_scores":  [round(s, 4) for s in scores.tolist()],
                "status":     "success",
            })
        except Exception as e:
            results.append({
                "model":  name,
                "score":  0.0,
                "std":    0.0,
                "metric": metric,
                "status": "failed",
                "error":  str(e),
            })

    # ── Sort by score ─────────────────────────────────────────────────────────
    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]

    # ── Recommendation ────────────────────────────────────────────────────────
    score = best["score"]
    if task_type == "classification":
        if score >= 0.90:
            rec = f"{best['model']} achieves {score:.1%} accuracy — excellent, ready for production."
        elif score >= 0.75:
            rec = f"{best['model']} achieves {score:.1%} accuracy — good, consider feature engineering."
        else:
            rec = f"Best model only achieves {score:.1%} — collect more data or engineer better features."
    else:
        if score >= 0.85:
            rec = f"{best['model']} achieves R² {score:.3f} — excellent fit."
        elif score >= 0.65:
            rec = f"{best['model']} achieves R² {score:.3f} — decent, try more features."
        else:
            rec = f"Best R² is {score:.3f} — consider non-linear models or more data."

    summary = (
        f"Tested {len(models)} models on {len(X)} rows. "
        f"Best: {best['model']} ({metric}={score:.4f} ± {best['std']:.4f})"
    )

    return {
        "task_type":      task_type,
        "target":         target_col,
        "best_model":     best["model"],
        "best_score":     best["score"],
        "best_std":       best["std"],
        "best_metric":    metric,
        "results":        results,
        "summary":        summary,
        "recommendation": rec,
        "n_rows":         len(X),
        "n_features":     len(X.columns),
    }