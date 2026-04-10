"""
correlation_network.py — Correlation Network Graph.

Builds a network of column relationships for visualization.
Each node = column, each edge = relationship strength.

Used to generate interactive network graphs in the web UI
and structured data for CLI output.
"""

from __future__ import annotations
import pandas as pd
import math
from typing import Any


def build_correlation_network(
    data:      dict[str, Any],
    threshold: float = 0.3,
) -> dict[str, Any]:
    """
    Build a correlation network from all column pairs.

    Args:
        data:      Structured data dict from loader.
        threshold: Minimum correlation strength to include as edge (0-1).

    Returns:
        {
            "nodes": list of node dicts,
            "edges": list of edge dicts,
            "summary": str,
            "strongest": list of top 5 pairs,
        }
    """
    df   = data["df"].copy()
    cols = list(df.columns)

    nodes: list[dict] = []
    edges: list[dict] = []

    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols if df[c].dtype == object]

    # ── Build nodes ───────────────────────────────────────────────────────────
    for col in cols:
        n_unique = df[col].nunique()
        nodes.append({
            "id":      col,
            "type":    "numeric" if col in num_cols else "categorical",
            "unique":  n_unique,
            "missing": int(df[col].isnull().sum()),
        })

    # ── Build edges — Numeric vs Numeric (Pearson) ────────────────────────────
    for i, c1 in enumerate(num_cols):
        for c2 in num_cols[i+1:]:
            pair = df[[c1, c2]].dropna()
            if len(pair) < 3:
                continue
            if pair[c1].std() == 0 or pair[c2].std() == 0:
                continue
            corr     = pair[c1].corr(pair[c2])
            strength = round(abs(corr), 4)
            if strength >= threshold:
                edges.append({
                    "source":    c1,
                    "target":    c2,
                    "strength":  strength,
                    "direction": "positive" if corr > 0 else "negative",
                    "method":    "pearson",
                    "weight":    round(strength * 10, 1),
                })

    # ── Build edges — Categorical vs Categorical (Cramér's V) ─────────────────
    for i, c1 in enumerate(cat_cols):
        for c2 in cat_cols[i+1:]:
            pair = df[[c1, c2]].dropna()
            if len(pair) < 3:
                continue
            v = _cramers_v(pair[c1], pair[c2])
            if v >= threshold:
                edges.append({
                    "source":    c1,
                    "target":    c2,
                    "strength":  v,
                    "direction": "association",
                    "method":    "cramers_v",
                    "weight":    round(v * 10, 1),
                })

    # ── Build edges — Numeric vs Categorical (Point-biserial) ─────────────────
    for cn in num_cols:
        for cc in cat_cols:
            pair = df[[cn, cc]].dropna()
            if len(pair) < 3:
                continue
            r = _point_biserial(pair[cn], pair[cc])
            if r >= threshold:
                edges.append({
                    "source":    cn,
                    "target":    cc,
                    "strength":  r,
                    "direction": "association",
                    "method":    "point_biserial",
                    "weight":    round(r * 10, 1),
                })

    # ── Sort edges by strength ────────────────────────────────────────────────
    edges.sort(key=lambda x: x["strength"], reverse=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_strong   = sum(1 for e in edges if e["strength"] >= 0.7)
    n_moderate = sum(1 for e in edges if 0.5 <= e["strength"] < 0.7)
    n_weak     = sum(1 for e in edges if e["strength"] < 0.5)

    if not edges:
        summary = "No significant relationships found between columns."
    else:
        parts = []
        if n_strong:   parts.append(f"{n_strong} strong")
        if n_moderate: parts.append(f"{n_moderate} moderate")
        if n_weak:     parts.append(f"{n_weak} weak")
        summary = f"Found {len(edges)} relationship(s): {', '.join(parts)}."

    return {
        "nodes":    nodes,
        "edges":    edges,
        "summary":  summary,
        "strongest": edges[:5],
        "stats": {
            "total_edges": len(edges),
            "strong":      n_strong,
            "moderate":    n_moderate,
            "weak":        n_weak,
        }
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    confusion = pd.crosstab(x, y)
    n = confusion.sum().sum()
    if n == 0:
        return 0.0
    chi2 = 0.0
    expected = (
        confusion.sum(axis=1).values[:, None] *
        confusion.sum(axis=0).values[None, :] / n
    )
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            e = expected[i][j]
            if e > 0:
                chi2 += (confusion.iloc[i, j] - e) ** 2 / e
    k = min(confusion.shape)
    if k <= 1 or n <= 1:
        return 0.0
    return round(math.sqrt(chi2 / (n * (k - 1))), 4)


def _point_biserial(numeric: pd.Series, categorical: pd.Series) -> float:
    cats = categorical.dropna().unique()
    if len(cats) != 2:
        return 0.0
    g1 = numeric[categorical == cats[0]].dropna()
    g2 = numeric[categorical == cats[1]].dropna()
    if len(g1) == 0 or len(g2) == 0:
        return 0.0
    n   = len(g1) + len(g2)
    std = numeric.std()
    if std == 0:
        return 0.0
    r = (g1.mean() - g2.mean()) / std * math.sqrt(len(g1) * len(g2) / (n * n))
    return round(abs(r), 4)