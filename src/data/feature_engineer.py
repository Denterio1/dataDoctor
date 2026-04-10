"""
feature_engineer.py — Auto Feature Engineering.

Automatically generates new features from existing columns:
    - Date columns    → year, month, day, day_of_week, is_weekend
    - Numeric columns → log transform, squared, interaction terms
    - Text columns    → text_length, word_count, unique_chars
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Any


def engineer_features(
    data: dict[str, Any],
    date_features:    bool = True,
    numeric_features: bool = True,
    text_features:    bool = True,
    interactions:     bool = True,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """
    Automatically generate new features from existing columns.

    Args:
        data:             Structured data dict from loader.
        date_features:    Extract year/month/day from date columns.
        numeric_features: Add log/squared transforms for numeric cols.
        text_features:    Add length/word count for text columns.
        interactions:     Add multiplication of numeric column pairs.

    Returns:
        (enriched_data, feature_log)
        feature_log = {"date": [...], "numeric": [...], "text": [...], "interaction": [...]}
    """
    df  = data["df"].copy()
    log = {"date": [], "numeric": [], "text": [], "interaction": []}

    # ── Date features ─────────────────────────────────────────────────────────
    if date_features:
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    parsed = pd.to_datetime(df[col], infer_datetime_format=True)
                    if parsed.notna().mean() >= 0.8:
                        df[col] = parsed
                except Exception:
                    pass

            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[f"{col}_year"]        = df[col].dt.year
                df[f"{col}_month"]       = df[col].dt.month
                df[f"{col}_day"]         = df[col].dt.day
                df[f"{col}_day_of_week"] = df[col].dt.dayofweek
                df[f"{col}_is_weekend"]  = (df[col].dt.dayofweek >= 5).astype(int)
                log["date"].extend([
                    f"{col}_year", f"{col}_month", f"{col}_day",
                    f"{col}_day_of_week", f"{col}_is_weekend"
                ])

    # ── Numeric features ──────────────────────────────────────────────────────
    if numeric_features:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        for col in num_cols:
            if df[col].min() > 0:
                df[f"{col}_log"] = np.log1p(df[col])
                log["numeric"].append(f"{col}_log")

            df[f"{col}_squared"] = df[col] ** 2
            log["numeric"].append(f"{col}_squared")

    # ── Interaction terms ─────────────────────────────────────────────────────
    if interactions:
        num_cols = [c for c in data["df"].columns if pd.api.types.is_numeric_dtype(data["df"][c])]
        pairs_done = 0
        for i, c1 in enumerate(num_cols):
            for c2 in num_cols[i+1:]:
                if pairs_done >= 5:
                    break
                new_col = f"{c1}_x_{c2}"
                df[new_col] = df[c1] * df[c2]
                log["interaction"].append(new_col)
                pairs_done += 1

    # ── Text features ─────────────────────────────────────────────────────────
    if text_features:
        text_cols = [c for c in data["df"].columns if data["df"][c].dtype == object]
        for col in text_cols:
            df[f"{col}_length"]     = data["df"][col].fillna("").str.len()
            df[f"{col}_word_count"] = data["df"][col].fillna("").str.split().str.len()
            log["text"].extend([f"{col}_length", f"{col}_word_count"])

    enriched = {**data, "df": df}
    return enriched, log