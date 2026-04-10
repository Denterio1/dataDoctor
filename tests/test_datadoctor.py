"""
test_datadoctor.py — Full test suite for dataDoctor.

Run with:
    pytest tests/ -v
"""

import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loader import load_csv
from src.data.analyzer import shape, missing_values, duplicate_rows, basic_stats, full_report
from src.data.analyzer import detect_outliers
from src.data.cleaner import handle_missing, remove_duplicates
from src.data.ml_readiness import ml_readiness
from src.data.relationships import detect_relationships
from src.data.preparator import prepare_for_ml
from src.data.drift import detect_drift
from src.core.agent import DataDoctor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_csv(tmp_path):
    """Create a temporary CSV file with known issues."""
    path = tmp_path / "test.csv"
    path.write_text(
        "name,age,salary,city\n"
        "Alice,25,3000,Algiers\n"
        "Bob,,4500,Oran\n"
        "Carol,31,,Algiers\n"
        "Dave,28,3800,\n"
        "Alice,25,3000,Algiers\n"  # duplicate
    )
    return str(path)


@pytest.fixture
def clean_csv(tmp_path):
    """Create a clean CSV with no issues."""
    path = tmp_path / "clean.csv"
    path.write_text(
        "product,price,quantity\n"
        "Laptop,999,5\n"
        "Phone,499,10\n"
        "Tablet,299,8\n"
        "Watch,199,15\n"
        "Keyboard,79,20\n"
    )
    return str(path)


@pytest.fixture
def sample_data(sample_csv):
    return load_csv(sample_csv)


@pytest.fixture
def clean_data(clean_csv):
    return load_csv(clean_csv)


# ── Loader tests ──────────────────────────────────────────────────────────────

class TestLoader:
    def test_load_csv_returns_dict(self, sample_csv):
        data = load_csv(sample_csv)
        assert isinstance(data, dict)
        assert "df" in data
        assert "columns" in data
        assert "source" in data

    def test_load_csv_correct_shape(self, sample_csv):
        data = load_csv(sample_csv)
        assert len(data["df"]) == 5
        assert len(data["columns"]) == 4

    def test_load_csv_columns(self, sample_csv):
        data = load_csv(sample_csv)
        assert data["columns"] == ["name", "age", "salary", "city"]

    def test_load_csv_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_csv("nonexistent_file.csv")


# ── Analyzer tests ────────────────────────────────────────────────────────────

class TestAnalyzer:
    def test_shape(self, sample_data):
        s = shape(sample_data)
        assert s["rows"] == 5
        assert s["columns"] == 4

    def test_missing_values(self, sample_data):
        mv = missing_values(sample_data)
        assert mv["age"] == 1
        assert mv["salary"] == 1
        assert mv["city"] == 1
        assert mv["name"] == 0

    def test_duplicate_rows(self, sample_data):
        dupes = duplicate_rows(sample_data)
        assert dupes == 1

    def test_no_duplicates(self, clean_data):
        dupes = duplicate_rows(clean_data)
        assert dupes == 0

    def test_basic_stats_numeric(self, clean_data):
        stats = basic_stats(clean_data)
        assert stats["price"]["type"] == "numeric"

        assert stats["price"]["min"] == 79.0
        assert stats["price"]["max"] == 999.0

    def test_basic_stats_text(self, sample_data):
        stats = basic_stats(sample_data)
        assert stats["name"]["type"] == "text"
        assert stats["name"]["unique"] == 4

    def test_full_report(self, sample_data):
        report = full_report(sample_data)
        assert "shape" in report
        assert "missing_values" in report
        assert "duplicate_rows" in report
        assert "column_stats" in report
        assert "outliers" in report

    def test_detect_outliers_clean(self, clean_data):
        outliers = detect_outliers(clean_data)
        assert isinstance(outliers, dict)

    def test_detect_outliers_with_outlier(self, tmp_path):
        path = tmp_path / "outlier.csv"
        path.write_text("value\n1\n2\n2\n3\n2\n2\n1000\n")
        data = load_csv(str(path))
        outliers = detect_outliers(data)
        assert "value" in outliers


# ── Cleaner tests ─────────────────────────────────────────────────────────────

class TestCleaner:
    def test_remove_duplicates(self, sample_data):
        cleaned, removed = remove_duplicates(sample_data)
        assert removed == 1
        assert len(cleaned["df"]) == 4

    def test_remove_duplicates_no_change(self, clean_data):
        cleaned, removed = remove_duplicates(clean_data)
        assert removed == 0
        assert len(cleaned["df"]) == len(clean_data["df"])

    def test_handle_missing_mean(self, sample_data):
        cleaned, changes = handle_missing(sample_data, strategy="mean")
        assert "age" in changes
        assert "salary" in changes
        assert cleaned["df"]["age"].isnull().sum() == 0
        assert cleaned["df"]["salary"].isnull().sum() == 0

    def test_handle_missing_drop(self, sample_data):
        before = len(sample_data["df"])
        cleaned, changes = handle_missing(sample_data, strategy="drop")
        assert len(cleaned["df"]) < before
        assert "rows_dropped" in changes

    def test_handle_missing_mode(self, sample_data):
        cleaned, changes = handle_missing(sample_data, strategy="mode")
        assert cleaned["df"].isnull().sum().sum() == 0

    def test_handle_missing_median(self, sample_data):
        cleaned, changes = handle_missing(sample_data, strategy="median")
        assert cleaned["df"]["age"].isnull().sum() == 0

    def test_original_not_mutated(self, sample_data):
        original_missing = sample_data["df"].isnull().sum().sum()
        handle_missing(sample_data, strategy="mean")
        assert sample_data["df"].isnull().sum().sum() == original_missing


# ── ML Readiness tests ────────────────────────────────────────────────────────

class TestMLReadiness:
    def test_returns_score(self, clean_data):
        outliers = detect_outliers(clean_data)
        result = ml_readiness(clean_data, outliers)
        assert "score" in result
        assert "grade" in result
        assert "checks" in result
        assert 0 <= result["score"] <= 100

    def test_grade_values(self, clean_data):
        outliers = detect_outliers(clean_data)
        result = ml_readiness(clean_data, outliers)
        assert result["grade"] in ["A", "B", "C", "D", "F"]

    def test_small_dataset_penalised(self, sample_data):
        outliers = detect_outliers(sample_data)
        result = ml_readiness(sample_data, outliers)
        size_check = next(c for c in result["checks"] if c["name"] == "Dataset size")
        assert size_check["status"] in ["warn", "fail"]


# ── Relationships tests ───────────────────────────────────────────────────────

class TestRelationships:
    def test_returns_list(self, clean_data):
        rels = detect_relationships(clean_data)
        assert isinstance(rels, list)

    def test_sorted_by_strength(self, clean_data):
        rels = detect_relationships(clean_data, threshold=0.0)
        if len(rels) >= 2:
            assert rels[0]["strength"] >= rels[1]["strength"]

    def test_relationship_keys(self, clean_data):
        rels = detect_relationships(clean_data, threshold=0.0)
        if rels:
            assert "col_a" in rels[0]
            assert "col_b" in rels[0]
            assert "strength" in rels[0]
            assert "method" in rels[0]


# ── Preparator tests ──────────────────────────────────────────────────────────

class TestPreparator:
    def test_prepare_returns_data_and_log(self, sample_data):
        prepared, log = prepare_for_ml(sample_data)
        assert "df" in prepared
        assert isinstance(log, dict)

    def test_no_missing_after_prepare(self, sample_data):
        prepared, _ = prepare_for_ml(sample_data, missing_strategy="mean")
        assert prepared["df"].isnull().sum().sum() == 0

    def test_no_duplicates_after_prepare(self, sample_data):
        prepared, log = prepare_for_ml(sample_data)
        assert log["duplicates_removed"] >= 0
        assert prepared["df"].duplicated().sum() == 0

    def test_all_numeric_after_prepare(self, sample_data):
        prepared, _ = prepare_for_ml(sample_data, encode=True, scale=True)
        for col in prepared["df"].columns:
            assert pd.api.types.is_numeric_dtype(prepared["df"][col])


# ── Drift tests ───────────────────────────────────────────────────────────────

class TestDrift:
    def test_no_drift_same_data(self, clean_data):
        result = detect_drift(clean_data, clean_data)
        assert result["severity"] == "none"

    def test_drift_detected(self, tmp_path):
        base_path = tmp_path / "base.csv"
        curr_path = tmp_path / "curr.csv"
        base_path.write_text("value\n1\n2\n3\n4\n5\n")
        curr_path.write_text("value\n100\n200\n300\n400\n500\n")
        base = load_csv(str(base_path))
        curr = load_csv(str(curr_path))
        result = detect_drift(base, curr)
        assert result["severity"] != "none"
        assert len(result["drifted_columns"]) > 0

    def test_new_column_detected(self, tmp_path):
        base_path = tmp_path / "base.csv"
        curr_path = tmp_path / "curr.csv"
        base_path.write_text("a,b\n1,2\n3,4\n")
        curr_path.write_text("a,b,c\n1,2,3\n4,5,6\n")
        base = load_csv(str(base_path))
        curr = load_csv(str(curr_path))
        result = detect_drift(base, curr)
        schema_drift = any(d["column"] == "schema" for d in result["drifted_columns"])
        assert schema_drift


# ── Agent tests ───────────────────────────────────────────────────────────────

class TestDataDoctor:
    def test_inspect_returns_dict(self, sample_csv):
        doctor = DataDoctor()
        result = doctor.inspect(sample_csv)
        assert "source" in result
        assert "raw_analysis" in result
        assert "cleaning_log" in result
        assert "clean_data" in result
        assert "summary" in result

    def test_inspect_summary_is_string(self, sample_csv):
        doctor = DataDoctor()
        result = doctor.inspect(sample_csv)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_inspect_cleans_duplicates(self, sample_csv):
        doctor = DataDoctor(remove_dupes=True)
        result = doctor.inspect(sample_csv)
        assert result["cleaning_log"].get("duplicates_removed", 0) >= 1

    def test_inspect_no_dedup(self, sample_csv):
        doctor = DataDoctor(remove_dupes=False)
        result = doctor.inspect(sample_csv)
        assert "duplicates_removed" not in result["cleaning_log"]

    def test_inspect_strategies(self, sample_csv):
        for strategy in ["mean", "median", "mode", "drop"]:
            doctor = DataDoctor(missing_strategy=strategy)
            result = doctor.inspect(sample_csv)
            assert result["clean_data"]["df"].isnull().sum().sum() == 0