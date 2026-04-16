"""
cognitive_dna.py — DataDNA v0.4.0 Core Structures

Defines the "Statistical Identity" of a dataset across four dimensions:
1. Statistical (Distributional/Entropy)
2. Structural (Schema/Missingness)
3. ML (Separability/Signal)
4. Temporal (Trend/Seasonality)
"""

from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Tuple

@dataclass(frozen=True)
class StatisticalDNA:
    entropy_vector:     Tuple[float, ...]
    moment_signatures:  Tuple[Tuple[float, ...], ...]
    distribution_types: Tuple[str, ...]
    correlation_hash:   str
    normality_scores:   Tuple[float, ...]
    mutual_info_matrix: Tuple[Tuple[float, ...], ...]

@dataclass(frozen=True)
class StructuralDNA:
    schema_hash:          str
    missing_mechanism:    str  # "NONE", "MCAR", "MAR", "MNAR"
    missing_pattern_hash: str
    cardinality_profile:  Tuple[int, ...]
    outlier_density:      float
    duplicate_ratio:      float
    shape_signature:      Tuple[int, int]  # (rows, cols)

@dataclass(frozen=True)
class MLDNA:
    feature_redundancy:  float
    target_signal:       str  # "strong", "medium", "weak", "none"
    separability_score:  float
    class_balance_score: float
    recommended_task:    str  # "classification", "regression"
    complexity_score:    float

@dataclass(frozen=True)
class TemporalDNA:
    has_temporal:      bool
    temporal_columns:  Tuple[str, ...]
    trend_detected:    bool
    seasonality_score: float
    time_gaps_uniform: bool

@dataclass
class DataDNA:
    statistical: StatisticalDNA
    structural:  StructuralDNA
    ml:          MLDNA
    temporal:    TemporalDNA
    dna_hash:    str
    created_at:  str
    source:      str
    version:     str = "0.4.0"
    personality: str = ""
    tags:        List[str] = field(default_factory=list)

    @property
    def short_hash(self) -> str:
        return self.dna_hash[:8]

@dataclass
class SimilarityReport:
    overall_similarity: float
    dimension_scores:   Dict[str, float]
    key_differences:    List[str]
    recommendation:     str

class DNASimilarityEngine:
    """Compares two DataDNA objects and returns a SimilarityReport."""
    
    def compare(self, dna1: DataDNA, dna2: DataDNA) -> SimilarityReport:
        # Simplified comparison for v0.4.0
        s_score = 1.0 if dna1.statistical.correlation_hash == dna2.statistical.correlation_hash else 0.5
        t_score = 1.0 if dna1.structural.schema_hash == dna2.structural.schema_hash else 0.7
        
        overall = (s_score + t_score) / 2.0
        
        diffs = []
        if dna1.structural.schema_hash != dna2.structural.schema_hash:
            diffs.append("Schema has changed")
        if dna1.ml.recommended_task != dna2.ml.recommended_task:
            diffs.append(f"Task mismatch: {dna1.ml.recommended_task} vs {dna2.ml.recommended_task}")
            
        return SimilarityReport(
            overall_similarity = overall,
            dimension_scores   = {"statistical": s_score, "structural": t_score},
            key_differences    = diffs,
            recommendation     = "Datasets are highly similar" if overall > 0.8 else "Datasets have drifted"
        )

class CognitiveDNA:
    """Legacy wrapper or factory for generating DataDNA (in development)."""
    def __init__(self, df: pd.DataFrame, target_col: str | None = None):
        # This would normally compute the DNA. 
        # For now, we'll provide a mock to allow the system to run.
        self.df = df
        self.target_col = target_col
        self.created_at = datetime.now().isoformat()
        self.dna_hash   = hashlib.md5(str(df.shape).encode()).hexdigest()
        self.personality = "Standard Dataset"
        self.tags = ["auto-generated"]
        
        self.statistical = StatisticalDNA((), (), (), "hash", (), ())
        self.structural = StructuralDNA("shash", "NONE", "phash", (), 0.0, 0.0, df.shape)
        self.ml = MLDNA(0.0, "medium", 0.5, 1.0, "classification", 0.5)
        self.temporal = TemporalDNA(False, (), False, 0.0, True)

    @property
    def short_hash(self) -> str: return self.dna_hash[:8]
