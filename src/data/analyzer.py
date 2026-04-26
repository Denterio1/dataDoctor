import pandas as pd
import math
from typing import Any


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_df(data: dict[str, Any]) -> pd.DataFrame:
    """Return a pandas DataFrame from data."""
    if "df" in data:
        return data["df"]
    rows = data.get("rows")
    if isinstance(rows, list):
        return pd.DataFrame(rows)
    return pd.DataFrame()


def _numeric_values(rows: list[dict], col: str) -> list[float]:
    result: list[float] = []
    for row in rows:
        val = row.get(col)
        if val is not None:
            try:
                fval = float(val)
                if math.isfinite(fval):
                    result.append(fval)
            except (ValueError, TypeError, OverflowError):
                pass
    return result


def _safe_float(val: Any) -> float | None:
    try:
        fval = float(val)
        return fval if math.isfinite(fval) else None
    except (ValueError, TypeError, OverflowError):
        return None


# ── public API ────────────────────────────────────────────────────────────────

def shape(data: dict[str, Any]) -> dict[str, int]:
    df = _get_df(data)
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
    }


def missing_values(data: dict[str, Any]) -> dict[str, int]:
    df = _get_df(data)
    counts = df.isna().sum().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def duplicate_rows(data: dict[str, Any]) -> int:
    df = _get_df(data)
    try:
        return int(df.duplicated().sum())
    except Exception:
        # Fallback for unhashable types (e.g. lists in columns)
        return int(df.astype(str).duplicated().sum())


def basic_stats(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}

    # Maintain original dictionary-based logic if rows/columns exist
    if "rows" in data and "columns" in data:
        rows = data["rows"]
        columns = data["columns"]

        for col in columns:
            non_null = [row[col] for row in rows if row.get(col) is not None]
            numeric = _numeric_values(rows, col)

            # Determine type based on 80% numeric threshold
            is_numeric = bool(numeric) and (len(numeric) / max(len(non_null), 1) >= 0.8)

            if is_numeric:
                mean_val = sum(numeric) / len(numeric) if numeric else None
                
                # Safe unique count for potentially unhashable types
                try:
                    unique_count = len(set(non_null))
                except TypeError:
                    unique_count = len(set(str(v) for v in non_null))

                stats[col] = {
                    "type":   "numeric",
                    "unique": unique_count,
                    "count":  len(numeric),
                    "min":    _safe_float(min(numeric)) if numeric else None,
                    "max":    _safe_float(max(numeric)) if numeric else None,
                    "mean":   _safe_float(mean_val) if mean_val is not None else None,
                }
            else:
                freq: dict[str, int] = {}
                for v in non_null:
                    s_v = str(v)
                    freq[s_v] = freq.get(s_v, 0) + 1
                
                top = max(freq, key=freq.get) if freq else None
                
                try:
                    unique_count = len(set(non_null))
                except TypeError:
                    unique_count = len(freq)

                stats[col] = {
                    "type":       "text",
                    "unique":     unique_count,
                    "count":      len(non_null),
                    "most_common": top,
                }

        return stats

    # Fallback to pandas-based logic
    df = _get_df(data)

    for col in df.columns:
        series = df[col].dropna()

        if series.empty:
            stats[col] = {"type": "unknown", "unique": 0, "count": 0}
            continue

        if series.dtype.kind in "biufc":
            stats[col] = {
                "type": "numeric",
                "unique": int(series.nunique()),
                "count": int(series.count()),
                "min": _safe_float(series.min()),
                "max": _safe_float(series.max()),
                "mean": _safe_float(series.mean()),
            }
        else:
            try:
                unique_count = int(series.nunique())
            except TypeError:
                unique_count = int(series.astype(str).nunique())

            mode = series.mode()
            stats[col] = {
                "type": "text",
                "unique": unique_count,
                "count": int(series.count()),
                "most_common": mode.iloc[0] if not mode.empty else None,
            }

    return stats


def detect_outliers(data: dict[str, Any]) -> dict[str, Any]:
    df = _get_df(data)
    result = {}

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)][col]

        if not outliers.empty:
            # Limit returned values to avoid bloating reports
            val_list = outliers.tolist()
            result[col] = {
                "count":  int(len(outliers)),
                "lower":  _safe_float(lower),
                "upper":  _safe_float(upper),
                "values": val_list[:100], 
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
