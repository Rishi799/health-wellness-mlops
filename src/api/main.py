"""FastAPI application for cardiometabolic risk scoring (Phase 2 Task D).

Provides endpoints:
- GET  /health      : System and model availability check (returns 503 if model unloaded).
- GET  /model-info  : Champion model provenance metadata and schema specs.
- POST /predict     : Single-patient cardiometabolic risk scoring with validation.

Monitoring (Phase 3 Task 1):
- UUID-based X-Request-ID middleware injected on every request.
- Request latency measured and logged via the ``health_api`` logger.
- Successful /predict responses trigger PredictionLogger (JSONL sink).
- Logging failures are fully isolated — they never affect prediction responses.
"""

import re
import time
import uuid
from contextlib import asynccontextmanager
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.api.monitoring import PredictionLogger
from src.api.predictor import predictor
from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionInput,
    PredictionResponse,
)

logger = logging.getLogger("health_api")

# ---------------------------------------------------------------------------
# Ensure health_api logs reach Docker stdout/stderr (StreamHandler).
# Guard against duplicate handlers when the module is re-imported or when
# Uvicorn reloads the application.
# ---------------------------------------------------------------------------
if not logger.handlers:
    _stream_handler = logging.StreamHandler()
    _stream_handler.setLevel(logging.INFO)
    _stream_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    )
    logger.addHandler(_stream_handler)
logger.setLevel(logging.INFO)

MODEL_UNAVAILABLE = "Model is unavailable"

_prediction_logger = PredictionLogger()

# ---------------------------------------------------------------------------
# Request-ID validation
# ---------------------------------------------------------------------------

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9\-_]{1,128}$")


def _validated_request_id(raw: str | None) -> str:
    """Return *raw* unchanged if it passes the allowlist; otherwise a new UUID v4.

    Rules (applied in order):
    - Missing or empty → generate UUID v4.
    - Longer than 128 characters → generate UUID v4.
    - Characters outside [A-Za-z0-9\\-_] → generate UUID v4.
    - Otherwise → return raw unchanged.
    """
    if raw and _REQUEST_ID_RE.match(raw):
        return raw
    return str(uuid.uuid4())


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
        "Educational awareness service using NHANES clinical, demographic, and lifestyle features. "
        "The model estimates association with prior self-reported diagnosis, not future disease. "
        "Scores are uncalibrated model outputs and bands are project-defined, not clinical thresholds. "
        "The current evaluation detected only 2 of 17 positive cases aged 20-39. "
        "A low score must not be used to dismiss symptoms or delay professional care."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request-ID + latency middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def request_id_and_latency_middleware(request: Request, call_next):
    """Attach a UUID request ID and measure per-request latency.

    - Reads ``X-Request-ID`` from the incoming headers; generates a UUID v4
      if the header is absent or blank.
    - Injects ``X-Request-ID`` into the response headers.
    - Logs method, path, status code, latency, and request ID via the
      ``health_api`` logger once the downstream handler returns.
    """
    request_id = _validated_request_id(request.headers.get("X-Request-ID"))
    # Stash on the request state so endpoint handlers can access it.
    request.state.request_id = request_id

    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000.0

    response.headers["X-Request-ID"] = request_id

    logger.info(
        "request completed - method=%s path=%s status=%d latency_ms=%.3f request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
        request_id,
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round(latency_ms, 3),
            "request_id": request_id,
        },
    )
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


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
                "error": MODEL_UNAVAILABLE,
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
            detail=MODEL_UNAVAILABLE,
        )
    return ModelInfoResponse(**{
        **predictor.model_info,
        "service_git_commit": os.getenv("SERVICE_GIT_COMMIT", "unknown"),
    })


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
def predict(payload: PredictionInput, request: Request) -> Any:
    """Perform cardiometabolic risk scoring on patient features.

    Validates inputs strictly against data contracts, preserves exact
    feature ordering, and stratifies risk into Low (<30), Moderate (30–59), or High (>=60).

    After a successful prediction the event is forwarded to PredictionLogger.
    Any logging failure is silently swallowed — it never alters the response.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=MODEL_UNAVAILABLE,
        )

    # Capture the start time for latency that is passed to PredictionLogger.
    # The middleware also measures end-to-end latency; this inner timer covers
    # only the inference path so the JSONL entry reflects model time, not I/O.
    t0 = time.perf_counter()
    try:
        response = predictor.predict(payload)
    except Exception:
        logger.exception("Unexpected inference error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction could not be completed",
        )
    inference_ms = (time.perf_counter() - t0) * 1000.0

    # Retrieve request ID injected by the middleware.
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    # Fire-and-forget logging — any exception is fully contained.
    try:
        _prediction_logger.log_prediction(
            input_data=payload,
            response=response,
            request_id=request_id,
            latency_ms=inference_ms,
        )
    except Exception as log_exc:  # pragma: no cover — PredictionLogger catches internally
        logger.warning("PredictionLogger raised unexpectedly: %s", log_exc)

    return response
