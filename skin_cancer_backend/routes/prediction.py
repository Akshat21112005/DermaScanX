"""
routes/prediction.py
---------------------
FastAPI router for all prediction-related endpoints.

Endpoints:
    POST  /api/v1/predict          – Upload an image and get a prediction
    GET   /api/v1/classes          – List all supported disease classes
    GET   /api/v1/model/info       – Model metadata / config info
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from controllers.prediction_controller import handle_predict, handle_get_classes
from schemas.prediction import ClassesResponse, ErrorResponse, PredictionResponse
from config.settings import settings
from utils.image_preprocessor import get_transform_description

logger = logging.getLogger(__name__)

router = APIRouter()


# ── POST /predict ─────────────────────────────────────────────────────────────
@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image file"},
        413: {"model": ErrorResponse, "description": "File too large"},
        503: {"model": ErrorResponse, "description": "Model not ready"},
    },
    summary="Classify a skin lesion image",
    description=(
        "Upload a JPEG/PNG/WebP/BMP image of a skin lesion. "
        "Returns the top predicted diagnosis with confidence scores."
    ),
)
async def predict(
    request: Request,
    file: UploadFile = File(
        ...,
        description="Skin lesion image (JPEG / PNG / WebP / BMP, max 10 MB)",
    ),
    top_k: int = Query(
        default=settings.TOP_K_PREDICTIONS,
        ge=1,
        le=7,
        description="Number of top predictions to return (1–7)",
    ),
) -> PredictionResponse:
    logger.info(
        f"POST /predict | file='{file.filename}' "
        f"content_type={file.content_type} top_k={top_k}"
    )
    return await handle_predict(request=request, file=file, top_k=top_k)


# ── GET /classes ──────────────────────────────────────────────────────────────
@router.get(
    "/classes",
    response_model=ClassesResponse,
    summary="List all supported skin lesion classes",
    description=(
        "Returns all 7 HAM10000 skin lesion classes that the model can classify, "
        "along with their dataset short codes."
    ),
)
def get_classes(request: Request) -> ClassesResponse:
    classes = handle_get_classes(request)
    return ClassesResponse(total_classes=len(classes), classes=classes)


# ── GET /model/info ───────────────────────────────────────────────────────────
@router.get(
    "/model/info",
    summary="Model and preprocessing metadata",
    description="Returns model architecture details and preprocessing pipeline info.",
)
def model_info(request: Request) -> JSONResponse:
    model_service = getattr(request.app.state, "model_service", None)
    return JSONResponse(
        content={
            "success": True,
            "architecture": "EfficientNet-B3",
            "framework": "timm + PyTorch",
            "dataset": "HAM10000",
            "num_classes": settings.NUM_CLASSES,
            "load_strategy": settings.MODEL_LOAD_STRATEGY,
            "model_loaded": model_service.is_loaded if model_service else False,
            "preprocessing": get_transform_description(),
            "artifacts": {
                "pkl_path":     str(settings.MODEL_PKL_PATH),
                "weights_path": str(settings.MODEL_WEIGHTS_PATH),
                "labels_path":  str(settings.CLASS_LABELS_PATH),
            },
        }
    )
