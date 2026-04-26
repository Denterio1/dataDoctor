"""
reliability.py - Robust ingestion and validation layer for tab flows.
"""

from __future__ import annotations

import io
import json
import os
from typing import Any

import pandas as pd


def _dedupe_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for col in columns:
        key = col.strip() or "unnamed"
        n = seen.get(key, 0)
        if n == 0:
            out.append(key)
        else:
            out.append(f"{key}_{n+1}")
        seen[key] = n + 1
    return out


def _read_csv_robust(raw: bytes, warnings: list[str]) -> pd.DataFrame:
    attempts = [
        ("utf-8-sig", None),
        ("utf-8", None),
        ("latin-1", None),
        ("utf-8-sig", ","),
        ("utf-8-sig", ";"),
        ("utf-8-sig", "\t"),
        ("utf-8-sig", "|"),
    ]

    last_error: Exception | None = None
    for encoding, sep in attempts:
        try:
            kwargs: dict[str, Any] = {
                "encoding": encoding,
                "on_bad_lines": "skip",
                "engine": "python",
            }
            if sep is None:
                kwargs["sep"] = None
                kwargs["delimiter"] = None
            else:
                kwargs["sep"] = sep
            df = pd.read_csv(io.BytesIO(raw), **kwargs)
            if not df.empty or len(df.columns) > 0:
                if encoding != "utf-8-sig":
                    warnings.append(f"Parsed CSV using fallback encoding '{encoding}'.")
                if sep not in (None, ","):
                    warnings.append(f"Detected non-standard CSV delimiter '{sep}'.")
                return df
        except Exception as exc:  # pragma: no cover - best effort fallback
            last_error = exc

    raise ValueError(f"Could not parse CSV with robust fallbacks: {last_error}")


def load_bytes_resilient(
    file_bytes: bytes,
    file_name: str,
    *,
    max_rows: int = 1_000_000,
    max_cols: int = 2_000,
    sample_rows: int = 250_000,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Load uploaded bytes with robust parsing and safety checks.
    Returns:
      {
        "data": {"columns": [...], "df": DataFrame, "source": str},
        "warnings": [...],
        "risk_flags": [...],
        "meta": {"sampled": bool, "rows_before": int, "rows_after": int}
      }
    """
    ext = os.path.splitext(file_name)[1].lower()
    warnings: list[str] = []
    risk_flags: list[str] = []

    if ext == ".csv":
        df = _read_csv_robust(file_bytes, warnings)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    elif ext == ".json":
        try:
            payload = json.loads(file_bytes.decode("utf-8", errors="replace"))
            if isinstance(payload, dict):
                df = pd.json_normalize(payload)
                warnings.append("JSON object detected; flattened with json_normalize.")
            elif isinstance(payload, list):
                df = pd.json_normalize(payload)
                warnings.append("JSON list detected; normalized to tabular form.")
            else:
                raise ValueError("Unsupported JSON structure for tabular conversion.")
        except json.JSONDecodeError:
            # Fallback for line-delimited/tabular JSON variants.
            df = pd.read_json(io.BytesIO(file_bytes))
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if len(df.columns) == 0:
        raise ValueError("No columns detected. The file may be malformed.")
    if len(df) == 0:
        raise ValueError("No rows detected. The file appears empty.")

    # Normalize column names for downstream consistency.
    df.columns = _dedupe_columns([str(c).strip() for c in df.columns])

    rows_before = len(df)
    sampled = False

    if len(df.columns) > max_cols:
        risk_flags.append(f"Very wide dataset ({len(df.columns)} columns). Some features may be slower.")
    if rows_before > max_rows:
        sampled = True
        df = df.sample(n=min(sample_rows, rows_before), random_state=random_state)
        warnings.append(
            f"Dataset is very large ({rows_before:,} rows). Sampled to {len(df):,} rows for interactive analysis."
        )
        risk_flags.append("Large dataset sampled for performance.")

    data = {"columns": list(df.columns), "df": df, "source": file_name}
    return {
        "data": data,
        "warnings": warnings,
        "risk_flags": risk_flags,
        "meta": {
            "sampled": sampled,
            "rows_before": rows_before,
            "rows_after": len(df),
            "columns": len(df.columns),
        },
    }
