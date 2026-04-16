from typing import Any


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_df(data: dict[str, Any]):
    """Return a pandas DataFrame from data."""
    if "df" in data:
        return data["df"]

    import pandas as pd
    return pd.DataFrame(data.get("rows", []))


def _numeric_values(rows: list[dict], col: str) -> list[float]:
    result: list[float] = []
    for row in rows:
        val = row.get(col)
        if val is not None:
            try:
                result.append(float(val))
            except (ValueError, TypeError):
                pass
    return result


# ── public API ────────────────────────────────────────────────────────────────

def shape(data: dict[str, Any]) -> dict[str, int]:
    df = _get_df(data)
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
    }


def missing_values(data: dict[str, Any]) -> dict[str, int]:
    df = _get_df(data)
    return df.isna().sum().to_dict()


def duplicate_rows(data: dict[str, Any]) -> int:
    df = _get_df(data)
    return int(df.duplicated().sum())


def basic_stats(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}

    # نحافظ على نفس المنطق القديم لو rows موجودة
    if "rows" in data and "columns" in data:
        rows = data["rows"]
        columns = data["columns"]

        for col in columns:
            non_null = [row[col] for row in rows if row.get(col) is not None]
            numeric = _numeric_values(rows, col)

            if numeric and len(numeric) / max(len(non_null), 1) >= 0.8:
                mean_val = sum(numeric) / len(numeric) if numeric else None
                stats[col] = {
                    "type":   "numeric",
                    "unique": len(set(non_null)),
                    "count":  len(numeric),
                    "min":    min(numeric),
                    "max":    max(numeric),
                    "mean":   round(mean_val, 4) if mean_val is not None else None,
                }
            else:
                freq: dict[str, int] = {}
                for v in non_null:
                    freq[str(v)] = freq.get(str(v), 0) + 1
                top = max(freq, key=freq.get) if freq else None
                stats[col] = {
                    "type":       "text",
                    "unique":     len(set(non_null)),
                    "count":      len(non_null),
                    "most_common": top,
                }

        return stats

    # fallback: pandas
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
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": round(float(series.mean()), 4),
            }
        else:
            mode = series.mode()
            stats[col] = {
                "type": "text",
                "unique": int(series.nunique()),
                "count": int(series.count()),
                "most_common": mode.iloc[0] if not mode.empty else None,
            }

    return stats


def full_report(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "shape":          shape(data),
        "missing_values": missing_values(data),
        "duplicate_rows": duplicate_rows(data),
        "column_stats":   basic_stats(data),
    }


def detect_outliers(data: dict) -> dict:
    df = _get_df(data)

    import pandas as pd
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
            result[col] = {
                "count":  int(len(outliers)),
                "lower":  round(float(lower), 4),
                "upper":  round(float(upper), 4),
                "values": outliers.tolist(),
            }

    return result