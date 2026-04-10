"""
cleaner.py — Data cleaning using pandas.
"""

import pandas as pd
from typing import Any, Literal

MissingStrategy = Literal["drop", "mean", "median", "mode", "fill"]


def remove_duplicates(data: dict[str, Any]) -> tuple[dict[str, Any], int]:
    df = data["df"].copy()
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    return {**data, "df": df}, removed


def handle_missing(
    data: dict[str, Any],
    strategy: MissingStrategy = "mean",
    fill_value: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    df = data["df"].copy()
    changes: dict[str, Any] = {}

    if strategy == "drop":
        before = len(df)
        df = df.dropna()
        changes["rows_dropped"] = before - len(df)
        return {**data, "df": df}, changes

    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        if null_count == 0:
            continue

        if strategy == "mean":
            if pd.api.types.is_numeric_dtype(df[col]):
                replacement = round(df[col].mean(), 4)
            else:
                replacement = df[col].mode()[0]

        elif strategy == "median":
            if pd.api.types.is_numeric_dtype(df[col]):
                replacement = df[col].median()
            else:
                replacement = df[col].mode()[0]

        elif strategy == "mode":
            replacement = df[col].mode()[0]

        elif strategy == "fill":
            replacement = fill_value

        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        df[col] = df[col].fillna(replacement)
        changes[col] = {
            "strategy":    strategy,
            "filled":      null_count,
            "replacement": replacement,
        }

    return {**data, "df": df}, changes