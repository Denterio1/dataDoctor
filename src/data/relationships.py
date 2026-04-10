"""
relationships.py — Detect relationships between columns.

Uses:
- Pearson correlation for numeric vs numeric
- Cramér's V for categorical vs categorical
- Point-biserial for numeric vs categorical
"""

from __future__ import annotations
import pandas as pd
import math
from typing import Any


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V association for two categorical columns."""
    confusion = pd.crosstab(x, y)
    n = confusion.sum().sum()
    if n == 0:
        return 0.0
    chi2 = 0.0
    expected = confusion.sum(axis=1).values[:, None] * confusion.sum(axis=0).values[None, :] / n
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
    """Simplified point-biserial correlation."""
    cats = categorical.dropna().unique()
    if len(cats) != 2:
        return 0.0
    g1 = numeric[categorical == cats[0]].dropna()
    g2 = numeric[categorical == cats[1]].dropna()
    if len(g1) == 0 or len(g2) == 0:
        return 0.0
    n = len(g1) + len(g2)
    std = numeric.std()
    if std == 0:
        return 0.0
    r = (g1.mean() - g2.mean()) / std * math.sqrt(len(g1) * len(g2) / (n * n))
    return round(abs(r), 4)


def detect_relationships(data: dict[str, Any], threshold: float = 0.5) -> list[dict]:
    """
    Detect relationships between all column pairs.

    Args:
        data:      Structured data dict from loader.
        threshold: Minimum strength to report (0-1).

    Returns:
        List of relationship dicts sorted by strength descending.
    """
    df      = data["df"].copy()
    cols    = df.columns.tolist()
    results = []

    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols if df[c].dtype == object]

    # Numeric vs Numeric — Pearson correlation
    for i, c1 in enumerate(num_cols):
        for c2 in num_cols[i+1:]:
            pair = df[[c1, c2]].dropna()
            if len(pair) < 3:
                continue
            std1 = pair[c1].std()
            std2 = pair[c2].std()
            if std1 == 0 or std2 == 0:
                continue
            corr = pair[c1].corr(pair[c2])
            strength = round(abs(corr), 4)
            if strength >= threshold:
                results.append({
                    "col_a":    c1,
                    "col_b":    c2,
                    "type":     "numeric-numeric",
                    "method":   "Pearson",
                    "strength": strength,
                    "direction": "positive" if corr > 0 else "negative",
                })

    # Categorical vs Categorical — Cramér's V
    for i, c1 in enumerate(cat_cols):
        for c2 in cat_cols[i+1:]:
            pair = df[[c1, c2]].dropna()
            if len(pair) < 3:
                continue
            v = _cramers_v(pair[c1], pair[c2])
            if v >= threshold:
                results.append({
                    "col_a":     c1,
                    "col_b":     c2,
                    "type":      "categorical-categorical",
                    "method":    "Cramér's V",
                    "strength":  v,
                    "direction": "association",
                })

    # Numeric vs Categorical — Point-biserial
    for cn in num_cols:
        for cc in cat_cols:
            pair = df[[cn, cc]].dropna()
            if len(pair) < 3:
                continue
            r = _point_biserial(pair[cn], pair[cc])
            if r >= threshold:
                results.append({
                    "col_a":     cn,
                    "col_b":     cc,
                    "type":      "numeric-categorical",
                    "method":    "Point-biserial",
                    "strength":  r,
                    "direction": "association",
                })

    results.sort(key=lambda x: x["strength"], reverse=True)
    return results