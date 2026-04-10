"""
agent.py — dataDoctor: the main autonomous data inspection agent.

Usage:
    from src.core.agent import DataDoctor

    doctor = DataDoctor()
    report = doctor.inspect("examples/sample_sales.csv")
    print(report["summary"])
"""

from __future__ import annotations

import textwrap
from typing import Any, Literal

from src.data.loader import load_csv
from src.data.analyzer import full_report
from src.data.cleaner import MissingStrategy, handle_missing, remove_duplicates


class DataDoctor:
    """Autonomous agent that inspects, cleans, and summarises a CSV file.

    Attributes:
        remove_dupes:      Whether to auto-remove duplicate rows.
        missing_strategy:  How to handle missing values
                           ("drop" | "mean" | "median" | "mode" | "fill").
        fill_value:        Used only when missing_strategy == "fill".
    """

    def __init__(
        self,
        remove_dupes: bool = True,
        missing_strategy: MissingStrategy = "mean",
        fill_value: Any = None,
    ) -> None:
        self.remove_dupes = remove_dupes
        self.missing_strategy = missing_strategy
        self.fill_value = fill_value

    # ── public API ────────────────────────────────────────────────────────────

    def inspect(self, filepath: str) -> dict[str, Any]:
        """Run the full inspection pipeline on *filepath*.

        Steps:
            1. Load the CSV.
            2. Analyse raw data quality.
            3. Optionally clean (remove dupes + handle missing).
            4. Build a human-readable text summary.

        Returns:
            {
                "source":        str,
                "raw_analysis":  dict,   # stats on the original file
                "cleaning_log":  dict,   # what the cleaner changed
                "clean_data":    dict,   # the cleaned data structure
                "summary":       str,    # human-readable report
            }
        """
        # 1. Load
        raw_data = load_csv(filepath)

        # 2. Analyse original data
        analysis = full_report(raw_data)

        # 3. Clean
        cleaning_log: dict[str, Any] = {}
        clean_data = raw_data

        if self.remove_dupes and analysis["duplicate_rows"] > 0:
            clean_data, dupes_removed = remove_duplicates(clean_data)
            cleaning_log["duplicates_removed"] = dupes_removed

        total_missing = sum(analysis["missing_values"].values())
        if total_missing > 0:
            clean_data, missing_changes = handle_missing(
                clean_data,
                strategy=self.missing_strategy,
                fill_value=self.fill_value,
            )
            cleaning_log["missing_values"] = missing_changes

        # 4. Build report
        summary = self._build_summary(filepath, analysis, cleaning_log)

        return {
            "source":       filepath,
            "raw_analysis": analysis,
            "cleaning_log": cleaning_log,
            "clean_data":   clean_data,
            "summary":      summary,
        }

    # ── internals ─────────────────────────────────────────────────────────────

    def _build_summary(
        self,
        filepath: str,
        analysis: dict[str, Any],
        cleaning_log: dict[str, Any],
    ) -> str:
        """Render a human-readable text report."""
        shape = analysis["shape"]
        missing = analysis["missing_values"]
        dupes = analysis["duplicate_rows"]
        stats = analysis["column_stats"]

        lines: list[str] = []

        # ── Header ──
        lines.append("=" * 60)
        lines.append("  dataDoctor — Inspection Report")
        lines.append("=" * 60)
        lines.append(f"  File   : {filepath}")
        lines.append(f"  Shape  : {shape['rows']} rows × {shape['columns']} columns")
        lines.append("")

        # ── Issues found ──
        lines.append("── Issues Found ──────────────────────────────────────────")
        issues_found = False

        if dupes > 0:
            lines.append(f"  ⚠  Duplicate rows   : {dupes}")
            issues_found = True

        missing_cols = {col: cnt for col, cnt in missing.items() if cnt > 0}
        if missing_cols:
            lines.append(f"  ⚠  Columns with missing values ({len(missing_cols)}):")
            for col, cnt in missing_cols.items():
                pct = cnt / shape["rows"] * 100
                lines.append(f"       • {col}: {cnt} missing ({pct:.1f}%)")
            issues_found = True

        if not issues_found:
            lines.append("  ✓  No issues detected — data looks clean!")

        lines.append("")

        # ── Cleaning actions ──
        lines.append("── Cleaning Actions ──────────────────────────────────────")
        if not cleaning_log:
            lines.append("  No cleaning was necessary.")
        else:
            if "duplicates_removed" in cleaning_log:
                lines.append(f"  ✓  Removed {cleaning_log['duplicates_removed']} duplicate row(s).")
            if "missing_values" in cleaning_log:
                mv = cleaning_log["missing_values"]
                if "rows_dropped" in mv:
                    lines.append(f"  ✓  Dropped {mv['rows_dropped']} row(s) with missing values.")
                else:
                    for col, info in mv.items():
                        lines.append(
                            f"  ✓  '{col}': filled {info['filled']} value(s) "
                            f"using {info['strategy']} → {info['replacement']}"
                        )
        lines.append("")

        # ── Column statistics ──
        lines.append("── Column Statistics ─────────────────────────────────────")
        for col, s in stats.items():
            if s["type"] == "numeric":
                lines.append(
                    f"  {col} [numeric]  "
                    f"min={s['min']}  max={s['max']}  mean={s['mean']}  "
                    f"unique={s['unique']}"
                )
            else:
                lines.append(
                    f"  {col} [text]     "
                    f"unique={s['unique']}  most_common='{s['most_common']}'"
                )
        lines.append("")

        # ── Suggestions ──
        lines.append("── Suggestions ───────────────────────────────────────────")
        suggestions: list[str] = []

        high_missing = [
            col for col, cnt in missing_cols.items()
            if cnt / shape["rows"] > 0.3
        ]
        if high_missing:
            suggestions.append(
                f"  • Columns {high_missing} have > 30 % missing values — "
                "consider dropping them or collecting better data."
            )

        low_unique = [
            col for col, s in stats.items()
            if s["type"] == "text" and s["unique"] == 1
        ]
        if low_unique:
            suggestions.append(
                f"  • Columns {low_unique} have only one unique value — "
                "they carry no information and can likely be removed."
            )

        high_unique_numeric = [
            col for col, s in stats.items()
            if s["type"] == "numeric" and s["unique"] == shape["rows"]
        ]
        if high_unique_numeric:
            suggestions.append(
                f"  • Columns {high_unique_numeric} have all unique values — "
                "they may be ID columns; verify they are not accidentally treated as features."
            )

        if not suggestions:
            suggestions.append("  • Data looks healthy — no major structural issues.")

        lines.extend(suggestions)
        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)