import pandas as pd
from src.data.analyzer import basic_stats

def test_unhashable_types():
    data = {
        "columns": ["a", "b"],
        "rows": [
            {"a": 1, "b": [1, 2]},
            {"a": 2, "b": [3, 4]},
            {"a": 3, "b": None}
        ]
    }
    try:
        stats = basic_stats(data)
        print("Stats:", stats)
    except Exception as e:
        print("Caught exception:", e)

if __name__ == "__main__":
    test_unhashable_types()
