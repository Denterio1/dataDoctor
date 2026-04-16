"""
api.py — dataDoctor REST API (FastAPI)

Exposes dataDoctor functionality as a REST API.
Run with: uvicorn api:app --reload

Endpoints:
    POST /inspect          — Full inspection report
    POST /clean            — Clean dataset
    POST /ml-readiness     — ML Readiness Score
    POST /dna              — Compute Data DNA
    POST /drift            — Detect drift between two datasets
    GET  /health           — Health check
    GET  /docs             — Auto-generated API docs
"""

from __future__ import annotations

import io
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

import pandas as pd

from src.data.loader        import load_file
from src.data.analyzer      import full_report, detect_outliers
from src.data.cleaner       import handle_missing, remove_duplicates
from src.data.ml_readiness  import ml_readiness
from src.data.preparator    import prepare_for_ml
from src.data.drift         import detect_drift
from src.data.cognitive_dna import compute_dna
from src.core.agent         import DataDoctor
from src.security           import security


# ── Models ────────────────────────────────────────────────────────────────────

class InspectRequest(BaseModel):
    strategy:     str  = Field("mean", description="Missing value strategy: mean|median|mode|drop")
    remove_dupes: bool = Field(True,   description="Remove duplicate rows")


class CleanRequest(BaseModel):
    strategy:     str  = Field("mean", description="Missing value strategy")
    remove_dupes: bool = Field(True,   description="Remove duplicate rows")
    encode:       bool = Field(False,  description="Label encode categorical columns")
    scale:        bool = Field(False,  description="StandardScale numeric columns")


class DNARequest(BaseModel):
    target_col: str | None = Field(None, description="Target column name for ML fingerprint")


class DriftRequest(BaseModel):
    pass


# ── App ───────────────────────────────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title       = "🩺 dataDoctor API",
        description = "Autonomous data inspection, cleaning, and ML preparation agent.",
        version     = "0.4.0",
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["*"],
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_uploaded(upload: UploadFile) -> dict:
        """Load an UploadFile into a data dict."""
        content  = upload.file.read()
        ext      = os.path.splitext(upload.filename)[1].lower()
        buf      = io.BytesIO(content)

        if ext == ".csv":
            df = pd.read_csv(buf, encoding="utf-8-sig")
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(buf, engine="openpyxl")
        elif ext == ".json":
            df = pd.read_json(buf)
        else:
            raise HTTPException(400, f"Unsupported file type: {ext}")

        return {"columns": list(df.columns), "df": df, "source": upload.filename}

    def _make_serializable(obj: Any) -> Any:
        """Make objects JSON-serializable."""
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serializable(i) for i in obj]
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj

    # ── Endpoints ─────────────────────────────────────────────────────────────

    @app.get("/health", tags=["System"])
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "version": "0.4.0", "service": "dataDoctor"}

    @app.get("/", tags=["System"])
    async def root():
        return {
            "name":    "dataDoctor API",
            "version": "0.4.0",
            "docs":    "/docs",
            "health":  "/health",
        }

    @app.post("/inspect", tags=["Analysis"])
    async def inspect(
        file:     UploadFile = File(..., description="CSV, Excel, or JSON file"),
        strategy: str        = "mean",
        remove_dupes: bool   = True,
    ):
        """
        Full inspection: quality report, statistics, and cleaning log.
        """
        try:
            data   = _load_uploaded(file)
            doctor = DataDoctor(remove_dupes=remove_dupes, missing_strategy=strategy)

            # We can't use doctor.inspect() directly since it expects filepath
            # Run analysis manually
            analysis = full_report(data)
            outliers = detect_outliers(data)
            ml       = ml_readiness(data, outliers)

            return JSONResponse(_make_serializable({
                "source":    file.filename,
                "shape":     analysis["shape"],
                "missing":   analysis["missing_values"],
                "duplicates":analysis["duplicate_rows"],
                "outliers":  outliers,
                "ml_score":  ml["score"],
                "ml_grade":  ml["grade"],
                "summary":   ml["summary"],
                "stats":     {
                    col: {k: v for k, v in s.items()}
                    for col, s in analysis["column_stats"].items()
                },
            }))
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.post("/clean", tags=["Cleaning"])
    async def clean(
        file:         UploadFile = File(...),
        strategy:     str        = "mean",
        remove_dupes: bool       = True,
        encode:       bool       = False,
        scale:        bool       = False,
    ):
        """
        Clean dataset and return cleaned CSV.
        """
        try:
            data = _load_uploaded(file)

            if remove_dupes:
                data, n_dupes = remove_duplicates(data)
            else:
                n_dupes = 0

            data, changes = handle_missing(data, strategy=strategy)

            if encode or scale:
                data, log = prepare_for_ml(data, missing_strategy=strategy, scale=scale, encode=encode)
            else:
                log = {}

            # Return cleaned CSV
            csv_bytes = data["df"].to_csv(index=False).encode("utf-8")
            filename  = os.path.splitext(file.filename)[0] + "_cleaned.csv"

            return StreamingResponse(
                io.BytesIO(csv_bytes),
                media_type = "text/csv",
                headers    = {"Content-Disposition": f"attachment; filename={filename}"}
            )
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.post("/ml-readiness", tags=["ML"])
    async def ml_readiness_check(file: UploadFile = File(...)):
        """
        Compute ML Readiness Score for the dataset.
        """
        try:
            data     = _load_uploaded(file)
            outliers = detect_outliers(data)
            result   = ml_readiness(data, outliers)

            return JSONResponse(_make_serializable({
                "score":   result["score"],
                "grade":   result["grade"],
                "summary": result["summary"],
                "checks":  result["checks"],
            }))
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.post("/dna", tags=["DNA"])
    async def compute_dataset_dna(
        file:       UploadFile = File(...),
        target_col: str | None = None,
    ):
        """
        Compute Cognitive Data DNA fingerprint.
        """
        try:
            data   = _load_uploaded(file)
            dna    = compute_dna(data, target_col=target_col)

            return JSONResponse(_make_serializable({
                "dna_hash":    dna.dna_hash,
                "short_hash":  dna.short_hash,
                "personality": dna.personality,
                "tags":        dna.tags,
                "statistical": {
                    "entropy_vector":    list(dna.statistical.entropy_vector[:10]),
                    "correlation_hash":  dna.statistical.correlation_hash,
                    "distribution_types":list(dna.statistical.distribution_types),
                },
                "structural": {
                    "schema_hash":       dna.structural.schema_hash,
                    "missing_mechanism": dna.structural.missing_mechanism,
                    "outlier_density":   dna.structural.outlier_density,
                    "duplicate_ratio":   dna.structural.duplicate_ratio,
                    "shape":             dna.structural.shape_signature,
                },
                "ml": {
                    "recommended_task":   dna.ml.recommended_task,
                    "target_signal":      dna.ml.target_signal,
                    "separability_score": dna.ml.separability_score,
                    "feature_redundancy": dna.ml.feature_redundancy,
                    "complexity_score":   dna.ml.complexity_score,
                },
                "temporal": {
                    "has_temporal":     dna.temporal.has_temporal,
                    "temporal_columns": list(dna.temporal.temporal_columns),
                    "trend_detected":   dna.temporal.trend_detected,
                    "seasonality_score":dna.temporal.seasonality_score,
                },
            }))
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.post("/drift", tags=["Analysis"])
    async def detect_data_drift(
        baseline: UploadFile = File(..., description="Baseline (reference) file"),
        current:  UploadFile = File(..., description="Current (new) file"),
    ):
        """
        Detect statistical drift between baseline and current dataset.
        """
        try:
            base_data = _load_uploaded(baseline)
            curr_data = _load_uploaded(current)
            result    = detect_drift(base_data, curr_data)

            return JSONResponse(_make_serializable(result))
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.get("/version", tags=["System"])
    async def get_version():
        """Get API version and capabilities."""
        return {
            "version":      "0.4.0",
            "capabilities": [
                "inspect", "clean", "ml-readiness",
                "dna", "drift", "health"
            ],
            "supported_formats": ["csv", "xlsx", "json"],
            "max_file_size_mb":  500,
        }

else:
    # FastAPI not available — provide helpful message
    print("FastAPI not installed. Run: pip install fastapi uvicorn")
    app = None


if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("Install FastAPI first: pip install fastapi uvicorn python-multipart")
    else:
        try:
            import uvicorn
            uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
        except ImportError:
            print("Install uvicorn: pip install uvicorn")