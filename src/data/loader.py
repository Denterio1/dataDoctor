"""
loader.py — Read CSV, Excel, and JSON files using pandas.
"""

import pandas as pd
import os
from typing import Any


def load_file(filepath: str) -> dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(filepath, encoding="utf-8-sig")
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(filepath, engine="openpyxl")
        elif ext == ".json":
            df = pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        raise ValueError(f"Could not read file: {e}")

    if df.empty:
        raise ValueError("File is empty or has no data rows.")

    return {
        "columns": list(df.columns),
        "df":      df,
        "source":  filepath,
    }


def load_csv(filepath: str) -> dict[str, Any]:
    """Backward-compatible alias for load_file."""
    return load_file(filepath)