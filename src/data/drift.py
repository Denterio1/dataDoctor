"""
drift.py — Data Drift Detector.

Compares two datasets (baseline vs current) and detects
statistical changes that could affect ML model performance.

Drift checks per column:
    Numeric  : mean shift, std shift, range change
    Categoric: new categories, missing categories, distribution shift
    Both     : missing rate change
"""

from __future__ import annotations
import pandas as pd
import math
from typing import Any


def detect_drift(
    baseline: dict[str, Any],
    current:  dict[str, Any],
    mean_threshold:    float = 0.1,
    missing_threshold: float = 0.05,
) -> dict[str, Any]:
    """
    Compare baseline dataset to current dataset and detect drift.

    Args:
        baseline:          Structured data dict (reference/old data).
        current:           Structured data dict (new data to check).
        mean_threshold:    Relative mean change to flag as drift (default 10%).
        missing_threshold: Absolute missing rate change to flag (default 5%).

    Returns:
        {
            "drifted_columns": list of drifted column dicts,
            "stable_columns":  list of stable column names,
            "summary":         str,
            "severity":        "none" | "low" | "medium" | "high",
        }
    """
    df_base = baseline["df"]
    df_curr = current["df"]

    drifted: list[dict] = []
    stable:  list[str]  = []

    common_cols = [c for c in df_base.columns if c in df_curr.columns]

    for col in common_cols:
        b = df_base[col]
        c = df_curr[col]
        issues: list[str] = []

        # ── Missing rate ──────────────────────────────────────────────────────
        b_miss = b.isnull().mean()
        c_miss = c.isnull().mean()
        delta_miss = abs(c_miss - b_miss)
        if delta_miss >= missing_threshold:
            issues.append(
                f"missing rate: {b_miss:.1%} → {c_miss:.1%} "
                f"(Δ {delta_miss:+.1%})"
            )

        # ── Numeric drift ─────────────────────────────────────────────────────
        if pd.api.types.is_numeric_dtype(b) and pd.api.types.is_numeric_dtype(c):
            b_clean = b.dropna()
            c_clean = c.dropna()

            if len(b_clean) > 0 and len(c_clean) > 0:
                b_mean, c_mean = b_clean.mean(), c_clean.mean()
                b_std,  c_std  = b_clean.std(),  c_clean.std()
                b_min,  c_min  = b_clean.min(),  c_clean.min()
                b_max,  c_max  = b_clean.max(),  c_clean.max()

                # Mean shift (relative)
                if b_mean != 0:
                    rel_mean = abs(c_mean - b_mean) / abs(b_mean)
                    if rel_mean >= mean_threshold:
                        issues.append(
                            f"mean shift: {b_mean:.4g} → {c_mean:.4g} "
                            f"({rel_mean:.1%} change)"
                        )

                # Std shift (relative)
                if b_std and b_std != 0:
                    rel_std = abs(c_std - b_std) / abs(b_std)
                    if rel_std >= 0.25:
                        issues.append(
                            f"std shift: {b_std:.4g} → {c_std:.4g} "
                            f"({rel_std:.1%} change)"
                        )

                # Range change
                if c_min < b_min:
                    issues.append(f"new minimum: {b_min:.4g} → {c_min:.4g}")
                if c_max > b_max:
                    issues.append(f"new maximum: {b_max:.4g} → {c_max:.4g}")

        # ── Categorical drift ──────────────────────────────────────────────────
        elif b.dtype == object and c.dtype == object:
            b_cats = set(b.dropna().unique())
            c_cats = set(c.dropna().unique())

            new_cats  = c_cats - b_cats
            gone_cats = b_cats - c_cats

            if new_cats:
                issues.append(f"new categories: {sorted(new_cats)[:5]}")
            if gone_cats:
                issues.append(f"missing categories: {sorted(gone_cats)[:5]}")

            # Distribution shift — top category changed?
            b_top = b.value_counts().index[0] if len(b.dropna()) > 0 else None
            c_top = c.value_counts().index[0] if len(c.dropna()) > 0 else None
            if b_top and c_top and b_top != c_top:
                issues.append(f"top category: '{b_top}' → '{c_top}'")

        # ── Classify ──────────────────────────────────────────────────────────
        if issues:
            severity = "high" if len(issues) >= 3 else "medium" if len(issues) == 2 else "low"
            drifted.append({
                "column":   col,
                "issues":   issues,
                "severity": severity,
            })
        else:
            stable.append(col)

    # ── New / removed columns ──────────────────────────────────────────────────
    new_cols  = [c for c in df_curr.columns if c not in df_base.columns]
    gone_cols = [c for c in df_base.columns if c not in df_curr.columns]

    if new_cols:
        drifted.append({
            "column":   "schema",
            "issues":   [f"new columns added: {new_cols}"],
            "severity": "high",
        })
    if gone_cols:
        drifted.append({
            "column":   "schema",
            "issues":   [f"columns removed: {gone_cols}"],
            "severity": "high",
        })

    # ── Overall severity ──────────────────────────────────────────────────────
    if not drifted:
        severity = "none"
        summary  = "No drift detected — datasets are statistically similar."
    else:
        high_count = sum(1 for d in drifted if d["severity"] == "high")
        if high_count >= 2 or len(drifted) >= len(common_cols) * 0.5:
            severity = "high"
            summary  = f"Severe drift in {len(drifted)} column(s) — retrain your model."
        elif high_count == 1 or len(drifted) >= 2:
            severity = "medium"
            summary  = f"Moderate drift in {len(drifted)} column(s) — monitor closely."
        else:
            severity = "low"
            summary  = f"Minor drift in {len(drifted)} column(s) — keep an eye on it."

    return {
        "drifted_columns": drifted,
        "stable_columns":  stable,
        "summary":         summary,
        "severity":        severity,
        "new_shape":       {"rows": len(df_curr), "cols": len(df_curr.columns)},
        "base_shape":      {"rows": len(df_base), "cols": len(df_base.columns)},
    }