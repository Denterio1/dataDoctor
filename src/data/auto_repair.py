"""
auto_repair.py - Conservative, one-click safe repairs.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def apply_safe_fixes(
    data: dict[str, Any],
    contract_report: dict[str, Any],
    *,
    missing_strategy: str = "median",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    df = data["df"].copy()
    actions: list[dict[str, Any]] = []

    # 1) Drop duplicate rows
    before_rows = len(df)
    df = df.drop_duplicates()
    if len(df) != before_rows:
        actions.append(
            {"action": "drop_duplicates", "before_rows": before_rows, "after_rows": len(df)}
        )

    # 2) Drop constant columns
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        actions.append({"action": "drop_constant_columns", "columns": constant_cols, "count": len(constant_cols)})

    # 3) Fill high-missing columns only if <= 60%; otherwise leave (safe)
    for v in contract_report.get("violations", []):
        if v.get("kind") != "missing" or not v.get("column"):
            continue
        col = v["column"]
        if col not in df.columns:
            continue
        miss_ratio = float(df[col].isna().mean())
        if miss_ratio > 0.60:
            actions.append({"action": "skip_high_missing_column", "column": col, "missing_ratio": miss_ratio})
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fill = float(df[col].median()) if missing_strategy == "median" else float(df[col].mean())
        else:
            mode = df[col].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else "unknown"
        before_missing = int(df[col].isna().sum())
        df[col] = df[col].fillna(fill)
        after_missing = int(df[col].isna().sum())
        if before_missing != after_missing:
            actions.append(
                {
                    "action": "fill_missing",
                    "column": col,
                    "strategy": missing_strategy if pd.api.types.is_numeric_dtype(df[col]) else "mode",
                    "filled": before_missing - after_missing,
                }
            )

    repaired = {"columns": list(df.columns), "df": df, "source": data.get("source", "unknown")}
    return repaired, actions
