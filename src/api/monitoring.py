"""Monitoring utilities for health API.

Provides:
- ``PredictionLogger``: thread-safe logger writing a JSON line per successful
  prediction (configured via ``PREDICTION_LOG_PATH``, defaulting to
  ``logs/predictions.jsonl``).
- ``APIMetrics`` / ``metrics``: Prometheus metrics collectors backed by an
  isolated ``CollectorRegistry`` for HTTP traffic, inference outcomes, and
  model metadata.

JSONL contract (Phase 3 handbook, page 3):
    Each line is a JSON object with top-level keys:
        timestamp, request_id, model_version, latency_ms,
        inputs   (object with all 21 feature keys),
        probability, risk_score, risk_band.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Dict, Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    GCCollector,
    Gauge,
    Histogram,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
)

from src.api.schemas import PredictionInput, PredictionResponse

_logger = logging.getLogger("health_api")


class PredictionLogger:
    """Singleton-style logger for prediction events.

    Directory creation and file opening are deferred until the first call to
    :meth:`log_prediction`.  Any filesystem error (permission denied, disk
    full, etc.) is caught and logged as a warning — it **never** prevents
    FastAPI from starting or a successful prediction from being returned.

    The :class:`threading.Lock` protects concurrent threads *within the same
    process*.  It does **not** guarantee cross-process safety.  With the
    default single-worker Uvicorn deployment this is sufficient.  Under
    multiple workers each process holds its own file handle, and because
    writes are append-only the logfile remains safe.
    """

    _instance: Optional["PredictionLogger"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "PredictionLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # Initialise only once — subsequent calls are no-ops because
        # attributes are already present.
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.file_lock = threading.Lock()
        log_path = os.getenv("PREDICTION_LOG_PATH", "logs/predictions.jsonl")
        self.log_path = Path(log_path)
        # File handle is opened lazily on first write.
        self._file: Optional[IO[str]] = None

    # ------------------------------------------------------------------
    # Lazy file initialisation
    # ------------------------------------------------------------------

    def _ensure_file(self) -> IO[str]:
        """Open the log file on first use, creating directories as needed.

        Raises on failure so the caller can catch and warn.
        """
        if self._file is None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.log_path.open("a", encoding="utf-8")
        return self._file

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_prediction(
        self,
        input_data: PredictionInput,
        response: PredictionResponse,
        request_id: str,
        latency_ms: float,
    ) -> None:
        """Write a JSON-L line for a successful prediction.

        ``input_data`` and ``response`` are Pydantic models.  The 21 input
        features are nested under an ``"inputs"`` key to match the shared
        contract on page 3 of the Phase 3 handbook.  Timestamps are
        timezone-aware UTC (ISO-8601 with ``+00:00`` suffix).

        Any exception during directory creation, file opening, or writing
        is caught and logged as a warning — the prediction response is
        never altered.
        """
        try:
            entry: Dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "model_version": response.model_version,
                "latency_ms": round(latency_ms, 3),
            }
            # Nest all 21 input features inside "inputs".
            entry["inputs"] = input_data.model_dump()
            # Add prediction results.
            entry.update(
                {
                    "probability": response.probability,
                    "risk_score": response.risk_score,
                    "risk_band": response.risk_band,
                }
            )
            line = json.dumps(entry, separators=(",", ":"))
            # Thread-safe write.
            with self.file_lock:
                fh = self._ensure_file()
                fh.write(line + "\n")
                fh.flush()
        except Exception as exc:
            _logger.warning("Failed to write prediction log: %s", exc)


# ---------------------------------------------------------------------------
# Prometheus API metrics
# ---------------------------------------------------------------------------

REQUEST_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
RISK_SCORE_BUCKETS = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
PROBABILITY_BUCKETS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def create_metrics_registry() -> CollectorRegistry:
    """Create an isolated Prometheus registry with standard runtime collectors."""
    registry = CollectorRegistry(auto_describe=True)
    ProcessCollector(registry=registry)
    PlatformCollector(registry=registry)
    GCCollector(registry=registry)
    return registry


class APIMetrics:
    """Container for the six Prometheus metrics exposed by the inference service."""

    def __init__(self, registry: Optional[CollectorRegistry] = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry(auto_describe=True)

        self.api_requests_total = Counter(
            "api_requests_total",
            "Total number of HTTP API requests",
            ["endpoint", "method", "status"],
            registry=self.registry,
        )
        self.api_request_latency_seconds = Histogram(
            "api_request_latency_seconds",
            "HTTP API request latency in seconds",
            ["endpoint"],
            buckets=REQUEST_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.predictions_total = Counter(
            "predictions_total",
            "Total number of successful predictions",
            ["risk_band"],
            registry=self.registry,
        )
        self.prediction_risk_score = Histogram(
            "prediction_risk_score",
            "Distribution of predicted cardiometabolic risk scores (0-100)",
            buckets=RISK_SCORE_BUCKETS,
            registry=self.registry,
        )
        self.prediction_probability = Histogram(
            "prediction_probability",
            "Distribution of predicted cardiometabolic risk probabilities (0.0-1.0)",
            buckets=PROBABILITY_BUCKETS,
            registry=self.registry,
        )
        self.model_info = Gauge(
            "model_info",
            "Metadata for the currently loaded champion model",
            ["model_name", "model_version", "git_commit"],
            registry=self.registry,
        )

    def set_model_info(
        self,
        model_name: str,
        model_version: str,
        git_commit: str,
    ) -> None:
        """Update model metadata gauge, clearing stale labelsets."""
        self.model_info.clear()
        self.model_info.labels(
            model_name=model_name,
            model_version=model_version,
            git_commit=git_commit,
        ).set(1)

    def set_model_unloaded(self) -> None:
        """Clear active model metadata when no model is loaded."""
        self.model_info.clear()

    def record_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        latency_seconds: float,
    ) -> None:
        """Record an HTTP request count and latency observation."""
        self.api_requests_total.labels(
            endpoint=endpoint,
            method=method,
            status=str(status),
        ).inc()
        self.api_request_latency_seconds.labels(
            endpoint=endpoint,
        ).observe(max(0.0, latency_seconds))

    def record_prediction(self, response: PredictionResponse) -> None:
        """Record metrics for a completed prediction."""
        self.predictions_total.labels(risk_band=response.risk_band).inc()
        self.prediction_risk_score.observe(response.risk_score)
        self.prediction_probability.observe(response.probability)

    def render(self) -> bytes:
        """Serialize all registered metrics in Prometheus text exposition format."""
        return generate_latest(self.registry)


METRICS_REGISTRY = create_metrics_registry()
metrics = APIMetrics(registry=METRICS_REGISTRY)
