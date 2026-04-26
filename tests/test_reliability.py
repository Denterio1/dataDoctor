import json

from src.data.reliability import load_bytes_resilient


def test_csv_latin1_fallback_loads():
    raw = "name;city\nAndre;Alger\nMeriem;Oran\n".encode("latin-1")
    out = load_bytes_resilient(raw, "sample.csv")
    assert out["data"]["df"].shape[0] == 2
    assert "name" in out["data"]["columns"]


def test_json_object_normalized():
    raw = json.dumps({"user": {"id": 1, "name": "A"}}).encode("utf-8")
    out = load_bytes_resilient(raw, "obj.json")
    assert out["data"]["df"].shape[0] == 1
    assert any(col.startswith("user") for col in out["data"]["columns"])


def test_large_dataset_is_sampled():
    rows = ["a,b"] + [f"{i},{i+1}" for i in range(2100)]
    raw = ("\n".join(rows)).encode("utf-8")
    out = load_bytes_resilient(raw, "big.csv", max_rows=1000, sample_rows=500)
    assert out["meta"]["sampled"] is True
    assert out["meta"]["rows_after"] == 500
