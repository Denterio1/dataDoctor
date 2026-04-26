import pandas as pd

from src.data.contracts import evaluate_contracts
from src.data.auto_repair import apply_safe_fixes


def _mk_data(df: pd.DataFrame) -> dict:
    return {"columns": list(df.columns), "df": df, "source": "test"}


def test_contract_detects_missing_duplicates_constant():
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "const": [5, 5, 5, 5],
            "name": ["a", "a", "b", None],
        }
    )
    report = evaluate_contracts(_mk_data(df), max_missing_ratio=0.2)
    kinds = {v["kind"] for v in report["violations"]}
    assert "duplicates" in kinds
    assert "constant" in kinds
    assert "missing" in kinds


def test_auto_repair_drops_dupes_and_constants_and_fills():
    df = pd.DataFrame(
        {
            "x": [1.0, 1.0, 2.0, None],
            "const": [9, 9, 9, 9],
            "cat": ["a", "a", None, "b"],
        }
    )
    data = _mk_data(df)
    report = evaluate_contracts(data, max_missing_ratio=0.1)
    repaired, actions = apply_safe_fixes(data, report, missing_strategy="median")
    assert len(repaired["df"]) <= len(df)
    assert "const" not in repaired["df"].columns
    assert repaired["df"]["x"].isna().sum() == 0
    assert repaired["df"]["cat"].isna().sum() == 0
    assert len(actions) >= 1
