"""
target_detector.py — Automatic Target Column Detection.

Analyses the dataset and suggests which column is most likely
the ML target, along with the recommended task type.

Scoring factors:
    - Position: last column gets bonus
    - Cardinality: few unique values → classification
    - Binary columns: strong classification signal
    - Name hints: 'target', 'label', 'class', 'output', 'y', 'result'
    - Numeric continuous: regression candidate
"""

from __future__ import annotations
import pandas as pd
from typing import Any


NAME_HINTS = {
    "classification": [
        "target", "label", "class", "category", "type", "status",
        "output", "result", "y", "flag", "churn", "fraud", "spam",
        "survived", "approved", "diagnosis", "outcome"
    ],
    "regression": [
        "price", "cost", "salary", "revenue", "sales", "amount",
        "value", "score", "rate", "total", "count", "quantity",
        "age", "weight", "height", "duration", "distance"
    ],
}


def detect_target(data: dict[str, Any]) -> dict[str, Any]:
    """
    Detect the most likely target column for ML.

    Args:
        data: Structured data dict from loader.

    Returns:
        {
            "recommended":  str (column name),
            "task_type":    "classification" | "regression" | "unknown",
            "confidence":   "high" | "medium" | "low",
            "reason":       str,
            "candidates":   list of dicts (all scored columns),
        }
    """
    df      = data["df"]
    cols    = list(df.columns)
    n_rows  = len(df)
    scores  = []

    for i, col in enumerate(cols):
        score    = 0
        signals  = []
        n_unique = df[col].nunique()
        dtype    = df[col].dtype
        col_low  = col.lower()

        # ── Position signal ───────────────────────────────────────────────────
        if i == len(cols) - 1:
            score   += 30
            signals.append("last column")
        elif i == 0:
            score   -= 20
            signals.append("first column (likely ID)")

        # ── Name hint signal ──────────────────────────────────────────────────
        for hint in NAME_HINTS["classification"]:
            if hint in col_low:
                score   += 40
                signals.append(f"name hint '{hint}' → classification")
                break
        for hint in NAME_HINTS["regression"]:
            if hint in col_low:
                score   += 35
                signals.append(f"name hint '{hint}' → regression")
                break

        # ── Binary column signal ──────────────────────────────────────────────
        if n_unique == 2:
            score   += 35
            signals.append("binary column → classification")

        # ── Cardinality signal ────────────────────────────────────────────────
        cardinality_ratio = n_unique / max(n_rows, 1)

        if n_unique <= 10:
            score   += 25
            signals.append(f"{n_unique} unique values → classification")
        elif cardinality_ratio >= 0.95 and pd.api.types.is_numeric_dtype(df[col]):
            score   -= 15
            signals.append("high cardinality numeric → likely feature")
        elif cardinality_ratio >= 0.95:
            score   -= 30
            signals.append("all-unique text → likely ID column")

        # ── Numeric continuous signal ─────────────────────────────────────────
        if pd.api.types.is_numeric_dtype(df[col]) and n_unique > 20:
            score   += 15
            signals.append("continuous numeric → regression candidate")

        # ── Missing values penalty ────────────────────────────────────────────
        missing_rate = df[col].isnull().mean()
        if missing_rate > 0.1:
            score   -= 20
            signals.append(f"{missing_rate:.0%} missing → penalised")

        scores.append({
            "column":   col,
            "score":    score,
            "signals":  signals,
            "n_unique": n_unique,
            "dtype":    str(dtype),
        })

    scores.sort(key=lambda x: x["score"], reverse=True)
    best = scores[0]

    # ── Determine task type ───────────────────────────────────────────────────
    col      = best["column"]
    n_unique = best["n_unique"]
    col_low  = col.lower()

    if n_unique == 2:
        task = "classification"
    elif n_unique <= 10:
        task = "classification"
    elif any(h in col_low for h in NAME_HINTS["regression"]):
        task = "regression"
    elif any(h in col_low for h in NAME_HINTS["classification"]):
        task = "classification"
    elif pd.api.types.is_numeric_dtype(df[col]) and n_unique > 20:
        task = "regression"
    else:
        task = "unknown"

    # ── Confidence ────────────────────────────────────────────────────────────
    gap = best["score"] - (scores[1]["score"] if len(scores) > 1 else 0)
    if gap >= 30 or best["score"] >= 60:
        confidence = "high"
    elif gap >= 15 or best["score"] >= 35:
        confidence = "medium"
    else:
        confidence = "low"

    reason = " | ".join(best["signals"]) if best["signals"] else "position-based guess"

    return {
        "recommended": col,
        "task_type":   task,
        "confidence":  confidence,
        "reason":      reason,
        "score":       best["score"],
        "candidates":  scores[:5],
    }