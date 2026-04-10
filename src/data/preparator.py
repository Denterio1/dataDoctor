"""
preparator.py — Prepare a dataset for ML in one step.

Steps:
    1. Remove duplicates
    2. Handle missing values
    3. Drop useless columns (constant or all-unique ID cols)
    4. Encode text columns (Label Encoding)
    5. Scale numeric columns (StandardScaler)
"""

from __future__ import annotations
import pandas as pd
from typing import Any


def prepare_for_ml(
    data: dict[str, Any],
    missing_strategy: str = "mean",
    scale: bool = True,
    encode: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Full ML preparation pipeline.

    Args:
        data:             Structured data dict from loader.
        missing_strategy: How to handle missing values.
        scale:            Whether to apply StandardScaler to numeric cols.
        encode:           Whether to Label Encode text cols.

    Returns:
        (prepared_data, log)
    """
    df  = data["df"].copy()
    log: dict[str, Any] = {}

    # ── Step 1: Remove duplicates ─────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    log["duplicates_removed"] = before - len(df)

    # ── Step 2: Handle missing values ─────────────────────────────────────────
    missing_log = {}
    for col in df.columns:
        n_null = int(df[col].isnull().sum())
        if n_null == 0:
            continue
        if missing_strategy == "drop":
            df = df.dropna()
            log["rows_dropped"] = before - len(df)
            break
        elif missing_strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
            val = round(df[col].mean(), 4)
        elif missing_strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
            val = df[col].median()
        else:
            val = df[col].mode()[0] if not df[col].mode().empty else "unknown"
        df[col] = df[col].fillna(val)
        missing_log[col] = {"filled": n_null, "value": val}
    log["missing_filled"] = missing_log

    # ── Step 3: Drop useless columns ──────────────────────────────────────────
    dropped = []
    for col in df.columns:
        n_unique = df[col].nunique()
        if n_unique <= 1:
            dropped.append((col, "constant"))
        elif n_unique == len(df) and df[col].dtype == object:
            dropped.append((col, "all-unique ID"))
    for col, reason in dropped:
        df = df.drop(columns=[col])
    log["columns_dropped"] = dropped

    # ── Step 4: Encode text columns ───────────────────────────────────────────
    encoded = {}
    if encode:
        for col in df.select_dtypes(include="object").columns:
            mapping = {v: i for i, v in enumerate(df[col].unique())}
            df[col] = df[col].map(mapping)
            encoded[col] = mapping
    log["encoded"] = encoded

    # ── Step 5: Scale numeric columns ─────────────────────────────────────────
    scaled = []
    if scale:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        for col in num_cols:
            mean = df[col].mean()
            std  = df[col].std()
            if std > 0:
                df[col] = ((df[col] - mean) / std).round(6)
                scaled.append(col)
    log["scaled"] = scaled

    log["final_shape"] = {"rows": len(df), "columns": len(df.columns)}

    prepared = {**data, "df": df}
    return prepared, log