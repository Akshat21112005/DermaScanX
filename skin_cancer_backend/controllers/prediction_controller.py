"""
controllers/prediction_controller.py
--------------------------------------
Contains business-logic that sits between the route handler and the
ModelService. Responsibilities:
    - Validate the uploaded file
    - Call ModelService.predict()
    - Map raw result dict → PredictionResponse schema
    - Handle and re-raise domain errors as appropriate HTTP exceptions
"""

from __future__ import annotations

import logging

from fastapi import UploadFile, HTTPException, Request, status

from schemas.prediction import PredictionEntry, PredictionResponse
from utils.file_validator import validate_image_upload
from config.settings import settings

logger = logging.getLogger(__name__)


async def handle_predict(
    request: Request,
    file: UploadFile,
    top_k: int,
) -> PredictionResponse:
    """
    Orchestrate a single image prediction request.

    Args:
        request : FastAPI Request (used to access app.state.model_service).
        file    : Uploaded image file.
        top_k   : Number of top predictions to include in the response.

    Returns:
        PredictionResponse schema instance.

    Raises:
        HTTPException 503 if the model is not loaded.
        HTTPException 400 for invalid file or preprocessing errors.
        HTTPException 500 for unexpected inference errors.
    """
    # ── 1. Guard: model must be loaded ────────────────────────────────────────
    model_service = getattr(request.app.state, "model_service", None)
    if model_service is None or not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model is not loaded. The service is still initialising "
                "or failed to load the weight file. Check server logs."
            ),
        )

    # ── 2. Validate & read file ───────────────────────────────────────────────
    image_bytes = await validate_image_upload(file)

    # ── 3. Run inference ──────────────────────────────────────────────────────
    try:
        result = model_service.predict(image_bytes=image_bytes, top_k=top_k)
    except ValueError as e:
        # Image decoding / preprocessing error
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image preprocessing failed: {e}",
        ) from e
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Unexpected inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed due to an unexpected error.",
        ) from e

    # ── 4. Build response ─────────────────────────────────────────────────────
    top_entries = [
        PredictionEntry(
            rank=p["rank"],
            label=p["label"],
            short_code=p["short_code"],
            confidence=p["confidence"],
            confidence_pct=p["confidence_pct"],
        )
        for p in result["top_predictions"]
    ]

    return PredictionResponse(
        success=True,
        filename=file.filename,
        predicted_class=result["predicted_class"],
        predicted_short_code=result["predicted_short_code"],
        confidence=result["confidence"],
        confidence_pct=result["confidence_pct"],
        top_predictions=top_entries,
        model_version=f"EfficientNet-B3 | HAM10000 | strategy={model_service.strategy}",
        inference_time_ms=result["inference_time_ms"],
    )


def handle_get_classes(request: Request):
    """Return the list of all supported classes."""
    model_service = getattr(request.app.state, "model_service", None)
    if model_service is None or not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )
    return model_service.get_class_info()
