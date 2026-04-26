"""
audit_trail.py - Lightweight repair audit summaries.
"""

from __future__ import annotations

from typing import Any


def build_repair_audit(
    source: str,
    before_shape: tuple[int, int],
    after_shape: tuple[int, int],
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source": source,
        "before_shape": {"rows": before_shape[0], "cols": before_shape[1]},
        "after_shape": {"rows": after_shape[0], "cols": after_shape[1]},
        "actions": actions,
        "action_count": len(actions),
    }
