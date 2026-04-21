"""
routes/health.py
----------------
Health check endpoint.
Used by load balancers, Docker HEALTHCHECK, and monitoring tools
to verify that the API is alive and the model is loaded.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from schemas.prediction import HealthResponse
from config.settings import settings

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API health check",
    description=(
        "Returns the current health status of the API including "
        "whether the ML model has been loaded successfully."
    ),
)
def health_check(request: Request) -> HealthResponse:
    model_service = getattr(request.app.state, "model_service", None)
    loaded = model_service.is_loaded if model_service else False

    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_strategy=settings.MODEL_LOAD_STRATEGY,
        app_version=settings.APP_VERSION,
        message=(
            "API is running and model is loaded."
            if loaded
            else "API is running but model is NOT loaded. Check server logs."
        ),
    )


@router.get(
    "/ping",
    summary="Simple liveness probe",
    description="Lightweight ping — returns 200 immediately without checking model state.",
)
def ping() -> JSONResponse:
    return JSONResponse(content={"status": "pong"})
