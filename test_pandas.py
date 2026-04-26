import pandas as pd
from src.data.analyzer import _get_df

def test_pandas_unhashable():
    data = {
        "columns": ["a", "b"],
        "rows": [
            {"a": 1, "b": [1, 2]},
            {"a": 2, "b": [3, 4]},
            {"a": 3, "b": None}
        ]
    }
    df = _get_df(data)
    col = "b"
    series = df[col].dropna()
    print(f"Dtype: {series.dtype}")
    try:
        print(f"Nunique: {series.nunique()}")
    except Exception as e:
        print(f"Nunique failed: {e}")
        
    try:
        mode = series.mode()
        print(f"Mode: {mode.iloc[0] if not mode.empty else None}")
    except Exception as e:
        print(f"Mode failed: {e}")

if __name__ == "__main__":
    test_pandas_unhashable()
