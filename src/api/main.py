"""FastAPI application for cardiometabolic risk scoring (Phase 2 Task D).

Provides endpoints:
- GET  /health      : System and model availability check (returns 503 if model unloaded).
- GET  /model-info  : Champion model provenance metadata and schema specs.
- POST /predict     : Single-patient cardiometabolic risk scoring with validation.
"""

from contextlib import asynccontextmanager
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from src.api.predictor import predictor
from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionInput,
    PredictionResponse,
)

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifact and metadata once during application startup."""
    try:
        predictor.load()
        logger.info(
            f"Successfully loaded champion model '{predictor.model_info.get('model_name')}' "
            f"version '{predictor.model_version}'."
        )
    except Exception as exc:
        logger.error(f"Failed to load model during application startup: {exc}")
    yield


app = FastAPI(
    title="Cardiometabolic Risk Assessment API",
    description=(
        "Production inference service serving champion Logistic Regression model "
        "for predicting cardiometabolic risk from NHANES clinical, demographic, and lifestyle features."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check and model status",
    responses={
        200: {"description": "Service is healthy and model is loaded"},
        503: {"description": "Service unavailable: model failed to load or is not ready"},
    },
)
def health() -> Any:
    """Return application health status, reporting 503 if model is not loaded."""
    if not predictor.is_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "model_version": None,
                "error": predictor.load_error or "Model is not loaded",
            },
        )
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_version=predictor.model_version,
        error=None,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model metadata and feature contract",
    responses={
        200: {"description": "Model metadata and contract specification"},
        503: {"description": "Service unavailable: model metadata not loaded"},
    },
)
def model_info() -> Any:
    """Return champion model metadata, provenance, metrics, and feature list."""
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=predictor.load_error or "Model is not loaded",
        )
    return ModelInfoResponse(**predictor.model_info)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict cardiometabolic risk for a patient",
    responses={
        200: {"description": "Successful prediction with risk score and band"},
        422: {"description": "Validation error: invalid feature type, value, or out-of-range"},
        503: {"description": "Service unavailable: model is not loaded"},
    },
)
def predict(payload: PredictionInput) -> Any:
    """Perform cardiometabolic risk scoring on patient features.

    Validates inputs strictly against data contracts, preserves exact
    feature ordering, and stratifies risk into Low (<30), Moderate (30–59), or High (>=60).
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=predictor.load_error or "Model is not loaded",
        )
    try:
        return predictor.predict(payload)
    except Exception as exc:
        logger.exception("Unexpected inference error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failure: {exc}",
        )

