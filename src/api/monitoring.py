"""Monitoring utilities for health API.

Provides a thread-safe logger that writes a JSON line per successful prediction.
The log location can be overridden with the ``PREDICTION_LOG_PATH`` environment
variable; otherwise it defaults to ``logs/predictions.jsonl`` relative to the
project root.

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
