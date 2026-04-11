"""
split_advisor.py — Smart Train/Test Split Advisor.

Analyses the dataset and recommends the optimal train/test split strategy:
    - Standard split (80/20, 70/30)
    - Stratified split (for imbalanced classification)
    - Time-series split (for temporal data)
    - K-Fold cross-validation recommendation
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Any


def advise_split(
    data:       dict[str, Any],
    target_col: str,
    task_type:  str = "auto",
) -> dict[str, Any]:
    """
    Recommend the best train/test split strategy.

    Args:
        data:       Structured data dict from loader.
        target_col: Name of the target column.
        task_type:  "classification", "regression", or "auto".

    Returns:
        {
            "strategy":       str,
            "test_size":      float,
            "cv_folds":       int,
            "stratify":       bool,
            "time_series":    bool,
            "warnings":       list,
            "recommendations":list,
            "code":           str (ready-to-use sklearn code),
        }
    """
    df      = data["df"].copy()
    n_rows  = len(df)
    warnings: list[str]       = []
    recs:     list[str]       = []

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y = df[target_col].copy()

    # ── Detect task type ──────────────────────────────────────────────────────
    if task_type == "auto":
        n_unique = y.nunique()
        task_type = "classification" if (y.dtype == object or n_unique <= 20) else "regression"

    # ── Detect time series ────────────────────────────────────────────────────
    date_cols = [
        col for col in df.columns
        if any(kw in col.lower() for kw in ["date", "time", "year", "month", "day", "timestamp"])
    ]
    is_time_series = len(date_cols) > 0

    # ── Dataset size analysis ─────────────────────────────────────────────────
    if n_rows < 50:
        test_size = 0.2
        cv_folds  = 3
        warnings.append(f"Very small dataset ({n_rows} rows) — results may be unreliable.")
        recs.append("Collect more data. Minimum 200 rows recommended for ML.")
    elif n_rows < 200:
        test_size = 0.2
        cv_folds  = 5
        warnings.append(f"Small dataset ({n_rows} rows) — use cross-validation for reliable estimates.")
        recs.append("Use 5-fold cross-validation instead of a single train/test split.")
    elif n_rows < 1000:
        test_size = 0.2
        cv_folds  = 5
        recs.append("Standard 80/20 split works well for this size.")
    elif n_rows < 10_000:
        test_size = 0.15
        cv_folds  = 5
        recs.append("Consider 85/15 split — enough test data with more training.")
    else:
        test_size = 0.1
        cv_folds  = 5
        recs.append("Large dataset — 90/10 split gives plenty of test samples.")

    # ── Class imbalance for classification ───────────────────────────────────
    stratify = False
    if task_type == "classification":
        class_counts = y.value_counts()
        min_class    = class_counts.min()
        max_class    = class_counts.max()
        imbalance_ratio = max_class / min_class if min_class > 0 else 999

        if imbalance_ratio > 3:
            stratify = True
            warnings.append(
                f"Class imbalance detected (ratio {imbalance_ratio:.1f}:1) — "
                "use stratified split to preserve class distribution."
            )
            recs.append("Use stratify=y in train_test_split to maintain class proportions.")

        if min_class < cv_folds:
            cv_folds = max(2, int(min_class))
            warnings.append(
                f"Smallest class has only {min_class} sample(s) — "
                f"reducing CV folds to {cv_folds}."
            )

    # ── Time series recommendation ────────────────────────────────────────────
    if is_time_series:
        warnings.append(
            f"Temporal column(s) detected: {date_cols} — "
            "random splits will cause data leakage!"
        )
        recs.append("Use TimeSeriesSplit instead of random train_test_split.")
        recs.append("Always split by time: train on past, test on future.")

    # ── Build sklearn code ────────────────────────────────────────────────────
    train_pct = int((1 - test_size) * 100)
    test_pct  = int(test_size * 100)

    if is_time_series:
        code = f"""from sklearn.model_selection import TimeSeriesSplit
import pandas as pd

# Sort by time first
df = df.sort_values('{date_cols[0] if date_cols else 'date'}')

X = df.drop(columns=['{target_col}'])
y = df['{target_col}']

# Time-aware cross-validation
tscv = TimeSeriesSplit(n_splits={cv_folds})
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
"""
    elif stratify:
        code = f"""from sklearn.model_selection import train_test_split, StratifiedKFold

X = df.drop(columns=['{target_col}'])
y = df['{target_col}']

# Stratified split — preserves class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size={test_size},
    random_state=42,
    stratify=y,        # <-- key for imbalanced data
)

# Cross-validation
skf = StratifiedKFold(n_splits={cv_folds}, shuffle=True, random_state=42)
"""
    else:
        code = f"""from sklearn.model_selection import train_test_split, KFold

X = df.drop(columns=['{target_col}'])
y = df['{target_col}']

# Standard split ({train_pct}/{test_pct})
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size={test_size},
    random_state=42,
)

# Cross-validation
kf = KFold(n_splits={cv_folds}, shuffle=True, random_state=42)
"""

    # ── Determine strategy name ───────────────────────────────────────────────
    if is_time_series:
        strategy = "Time Series Split"
    elif stratify:
        strategy = "Stratified Split"
    else:
        strategy = "Standard Split"

    return {
        "strategy":        strategy,
        "test_size":       test_size,
        "train_size":      round(1 - test_size, 2),
        "cv_folds":        cv_folds,
        "stratify":        stratify,
        "time_series":     is_time_series,
        "task_type":       task_type,
        "n_rows":          n_rows,
        "warnings":        warnings,
        "recommendations": recs,
        "code":            code,
    }