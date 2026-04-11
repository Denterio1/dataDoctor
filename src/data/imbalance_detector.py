"""
imbalance_detector.py — Advanced Class Imbalance Detector.

Detects class imbalance and recommends the best handling strategy:
    - SMOTE (Synthetic Minority Over-sampling)
    - ADASYN (Adaptive Synthetic Sampling)
    - Random Under-sampling
    - Class weights adjustment
    - Combination methods (SMOTETomek, SMOTEENN)

Also provides:
    - Imbalance severity score
    - Impact on ML metrics
    - Ready-to-use code for each strategy
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Any


def detect_imbalance(
    data:       dict[str, Any],
    target_col: str,
) -> dict[str, Any]:
    """
    Detect class imbalance and recommend handling strategies.

    Args:
        data:       Structured data dict from loader.
        target_col: Name of the target column (must be categorical).

    Returns:
        {
            "is_imbalanced":  bool,
            "severity":       "none" | "mild" | "moderate" | "severe" | "extreme",
            "imbalance_ratio":float,
            "class_dist":     dict of class counts and percentages,
            "recommended":    str (best strategy),
            "strategies":     list of strategy dicts,
            "warnings":       list of str,
            "code":           dict of ready-to-use code per strategy,
        }
    """
    df = data["df"].copy()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y             = df[target_col].dropna()
    class_counts  = y.value_counts()
    n_classes     = len(class_counts)
    n_total       = len(y)
    majority      = class_counts.max()
    minority      = class_counts.min()
    ratio         = round(majority / minority, 2) if minority > 0 else 999.0

    # ── Class distribution ────────────────────────────────────────────────────
    class_dist = {}
    for cls, cnt in class_counts.items():
        class_dist[str(cls)] = {
            "count": int(cnt),
            "pct":   round(cnt / n_total * 100, 2),
        }

    # ── Severity ──────────────────────────────────────────────────────────────
    if ratio < 1.5:
        severity      = "none"
        is_imbalanced = False
    elif ratio < 3:
        severity      = "mild"
        is_imbalanced = True
    elif ratio < 10:
        severity      = "moderate"
        is_imbalanced = True
    elif ratio < 100:
        severity      = "severe"
        is_imbalanced = True
    else:
        severity      = "extreme"
        is_imbalanced = True

    warnings: list[str] = []

    if is_imbalanced:
        warnings.append(
            f"Imbalance ratio {ratio}:1 — minority class has only "
            f"{minority} samples ({minority/n_total*100:.1f}%)."
        )
    if minority < 10:
        warnings.append(
            f"Minority class has only {minority} samples — SMOTE requires at least 6."
        )
    if n_total < 100:
        warnings.append("Small dataset — oversampling may cause overfitting.")

    # ── Strategies ────────────────────────────────────────────────────────────
    strategies = []

    if minority >= 6:
        strategies.append({
            "name":        "SMOTE",
            "description": "Synthetic Minority Over-sampling Technique — generates synthetic samples",
            "best_for":    "moderate to severe imbalance with enough minority samples",
            "pros":        ["No data loss", "Creates realistic synthetic samples"],
            "cons":        ["Can create noisy samples", "Slower than random methods"],
            "recommended": ratio >= 3 and minority >= 6,
        })

        strategies.append({
            "name":        "ADASYN",
            "description": "Adaptive Synthetic Sampling — focuses on hard-to-classify minority samples",
            "best_for":    "when minority samples are hard to classify",
            "pros":        ["Focuses on difficult regions", "Adaptive generation"],
            "cons":        ["May amplify noise", "More complex"],
            "recommended": ratio >= 5 and minority >= 6,
        })

        strategies.append({
            "name":        "SMOTETomek",
            "description": "SMOTE + Tomek Links — oversample minority then clean boundaries",
            "best_for":    "severe imbalance with noisy boundaries",
            "pros":        ["Cleaner decision boundary", "Reduces noise"],
            "cons":        ["Slower", "Removes some majority samples"],
            "recommended": ratio >= 10 and minority >= 6,
        })

    strategies.append({
        "name":        "Class Weights",
        "description": "Adjust model's class weights without changing data",
        "best_for":    "when you cannot modify the dataset",
        "pros":        ["Simple", "No data modification", "Fast"],
        "cons":        ["Less effective for extreme imbalance"],
        "recommended": ratio < 10 or minority < 6,
    })

    strategies.append({
        "name":        "Random Under-sampling",
        "description": "Randomly remove majority class samples",
        "best_for":    "large datasets where losing data is acceptable",
        "pros":        ["Fast", "Simple", "Reduces training time"],
        "cons":        ["Loses potentially useful data"],
        "recommended": n_total > 1000 and ratio < 20,
    })

    # ── Best recommendation ───────────────────────────────────────────────────
    if not is_imbalanced:
        recommended = "No action needed"
    elif minority < 6:
        recommended = "Class Weights"
    elif ratio >= 10:
        recommended = "SMOTETomek"
    elif ratio >= 3:
        recommended = "SMOTE"
    else:
        recommended = "Class Weights"

    # ── Code snippets ─────────────────────────────────────────────────────────
    code = {}

    code["SMOTE"] = f"""from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X = df.drop(columns=['{target_col}'])
y = df['{target_col}']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Before: {{y_train.value_counts().to_dict()}}")
print(f"After:  {{pd.Series(y_train_resampled).value_counts().to_dict()}}")"""

    code["ADASYN"] = f"""from imblearn.over_sampling import ADASYN

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)"""

    code["SMOTETomek"] = f"""from imblearn.combine import SMOTETomek

smotetomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smotetomek.fit_resample(X_train, y_train)"""

    code["Class Weights"] = f"""from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes    = np.unique(y_train)
weights    = compute_class_weight('balanced', classes=classes, y=y_train)
class_dict = dict(zip(classes, weights))

# Use in any sklearn model:
model = RandomForestClassifier(class_weight=class_dict, random_state=42)
# Or use class_weight='balanced' directly:
model = RandomForestClassifier(class_weight='balanced', random_state=42)"""

    code["Random Under-sampling"] = f"""from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)"""

    return {
        "is_imbalanced":   is_imbalanced,
        "severity":        severity,
        "imbalance_ratio": ratio,
        "n_classes":       n_classes,
        "n_total":         n_total,
        "majority_class":  str(class_counts.index[0]),
        "minority_class":  str(class_counts.index[-1]),
        "class_dist":      class_dist,
        "recommended":     recommended,
        "strategies":      strategies,
        "warnings":        warnings,
        "code":            code,
    }