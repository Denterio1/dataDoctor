"""
analyzer.py — Data quality checks and statistics using pandas.
"""

import pandas as pd
from typing import Any


def shape(data: dict[str, Any]) -> dict[str, int]:
    df = data["df"]
    return {"rows": len(df), "columns": len(df.columns)}


def missing_values(data: dict[str, Any]) -> dict[str, int]:
    df = data["df"]
    return df.isnull().sum().to_dict()


def duplicate_rows(data: dict[str, Any]) -> int:
    df = data["df"]
    return int(df.duplicated().sum())


def basic_stats(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    df = data["df"]
    stats = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            stats[col] = {
                "type":   "numeric",
                "count":  int(df[col].count()),
                "unique": int(df[col].nunique()),
                "min":    round(float(df[col].min()), 4),
                "max":    round(float(df[col].max()), 4),
                "mean":   round(float(df[col].mean()), 4),
            }
        else:
            top = df[col].mode()[0] if not df[col].mode().empty else None
            stats[col] = {
                "type":        "text",
                "count":       int(df[col].count()),
                "unique":      int(df[col].nunique()),
                "most_common": top,
            }
    return stats


def detect_outliers(data: dict[str, Any]) -> dict[str, Any]:
    """Detect outliers in numeric columns using IQR method."""
    df     = data["df"]
    result = {}
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        Q1       = df[col].quantile(0.25)
        Q3       = df[col].quantile(0.75)
        IQR      = Q3 - Q1
        lower    = Q1 - 1.5 * IQR
        upper    = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)][col]
        if not outliers.empty:
            result[col] = {
                "count":  len(outliers),
                "lower":  round(lower, 4),
                "upper":  round(upper, 4),
                "values": outliers.tolist(),
            }
    return result


def full_report(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "shape":          shape(data),
        "missing_values": missing_values(data),
        "duplicate_rows": duplicate_rows(data),
        "column_stats":   basic_stats(data),
        "outliers":       detect_outliers(data),
    }