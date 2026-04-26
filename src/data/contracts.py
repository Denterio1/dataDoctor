"""
contracts.py - Dataset contract evaluation for existing UI flows.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def evaluate_contracts(
    data: dict[str, Any],
    *,
    max_missing_ratio: float = 0.35,
    max_unique_text_ratio: float = 0.95,
) -> dict[str, Any]:
    df = data["df"]
    rows = max(len(df), 1)
    violations: list[dict[str, Any]] = []

    # Dataset-level checks
    if len(df.columns) == 0:
        violations.append(
            {"level": "high", "kind": "schema", "column": None, "message": "No columns detected."}
        )
    if len(df) == 0:
        violations.append(
            {"level": "high", "kind": "schema", "column": None, "message": "No rows detected."}
        )

    # Column-level checks
    for col in df.columns:
        series = df[col]
        miss_ratio = float(series.isna().mean())
        if miss_ratio > max_missing_ratio:
            violations.append(
                {
                    "level": "high" if miss_ratio > 0.6 else "medium",
                    "kind": "missing",
                    "column": str(col),
                    "message": f"Missing ratio is {miss_ratio:.1%} (threshold {max_missing_ratio:.0%}).",
                }
            )

        uniq = int(series.nunique(dropna=False))
        if uniq <= 1:
            violations.append(
                {
                    "level": "medium",
                    "kind": "constant",
                    "column": str(col),
                    "message": "Column has one unique value (constant feature).",
                }
            )

        if series.dtype == object:
            uniq_ratio = uniq / rows
            if uniq_ratio >= max_unique_text_ratio and uniq > 25:
                violations.append(
                    {
                        "level": "medium",
                        "kind": "high_cardinality_text",
                        "column": str(col),
                        "message": f"High-cardinality text ({uniq_ratio:.1%} unique ratio).",
                    }
                )

    dupes = int(df.duplicated().sum())
    if dupes > 0:
        violations.append(
            {
                "level": "medium",
                "kind": "duplicates",
                "column": None,
                "message": f"{dupes} duplicate rows detected.",
            }
        )

    high = sum(1 for v in violations if v["level"] == "high")
    medium = sum(1 for v in violations if v["level"] == "medium")
    overall = "high" if high > 0 else "medium" if medium > 0 else "low"

    return {
        "overall_risk": overall,
        "violations": violations,
        "counts": {"high": high, "medium": medium, "total": len(violations)},
    }
