"""
schemas/prediction.py
---------------------
Pydantic v2 models that define the exact shape of every API
request and response. FastAPI uses these for automatic validation
and OpenAPI schema generation.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


# ── Individual prediction entry ───────────────────────────────────────────────
class PredictionEntry(BaseModel):
    """One class with its confidence score."""

    rank: int = Field(..., ge=1, description="Rank (1 = highest confidence)")
    label: str = Field(..., description="Human-readable class label")
    short_code: str = Field(..., description="Short code used in the dataset (e.g. 'mel')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax probability [0, 1]")
    confidence_pct: str = Field(..., description="Confidence as formatted percentage string")


# ── Top-level prediction response ─────────────────────────────────────────────
class PredictionResponse(BaseModel):
    """Returned by POST /predict."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "filename": "lesion_001.jpg",
                "predicted_class": "Melanoma",
                "predicted_short_code": "mel",
                "confidence": 0.8731,
                "confidence_pct": "87.31%",
                "top_predictions": [
                    {
                        "rank": 1,
                        "label": "Melanoma",
                        "short_code": "mel",
                        "confidence": 0.8731,
                        "confidence_pct": "87.31%",
                    },
                    {
                        "rank": 2,
                        "label": "Melanocytic nevi",
                        "short_code": "nv",
                        "confidence": 0.0821,
                        "confidence_pct": "8.21%",
                    },
                ],
                "model_version": "EfficientNet-B3 (HAM10000)",
                "inference_time_ms": 42.5,
            }
        }
    )

    success: bool
    filename: Optional[str] = None
    predicted_class: str
    predicted_short_code: str
    confidence: float
    confidence_pct: str
    top_predictions: List[PredictionEntry]
    model_version: str
    inference_time_ms: float


# ── Health check response ──────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' | 'degraded' | 'error'")
    model_loaded: bool
    model_strategy: str
    app_version: str
    message: str


# ── Error response ─────────────────────────────────────────────────────────────
class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None


# ── Classes list response ──────────────────────────────────────────────────────
class ClassesResponse(BaseModel):
    total_classes: int
    classes: List[dict]
