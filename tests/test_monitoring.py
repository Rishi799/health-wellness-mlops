"""Tests for PredictionLogger (Phase 3 monitoring contract).

Covers:
- Nested ``"inputs"`` object containing all 21 feature keys.
- Missing optional values serialised as JSON ``null``.
- Exact JSONL contract (top-level keys, nested structure, UTC timestamp).
- Directory-creation / file-opening failure (lazy init).
- Write failure without breaking prediction logging integration.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.api.schemas import PredictionInput, PredictionResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The 21 features defined by the data contract (order matches schemas.py).
ALL_21_FEATURES = [
    "age",
    "sex",
    "education",
    "income_ratio",
    "height_cm",
    "weight_kg",
    "bmi",
    "waist_cm",
    "sbp",
    "dbp",
    "pulse",
    "told_high_bp",
    "total_chol",
    "hdl",
    "smoking_status",
    "alcohol_drinks_week",
    "exercise_min_week",
    "sedentary_min_day",
    "sleep_hours",
    "phq9_score",
    "phq9_category",
]


def _full_input() -> PredictionInput:
    """Return a PredictionInput with every feature populated."""
    return PredictionInput(
        age=45,
        sex=1,
        education=3.0,
        income_ratio=2.5,
        height_cm=175.0,
        weight_kg=78.5,
        bmi=25.6,
        waist_cm=88.0,
        sbp=120.0,
        dbp=80.0,
        pulse=72.0,
        told_high_bp=0.0,
        total_chol=195.0,
        hdl=55.0,
        smoking_status="never",
        alcohol_drinks_week=2.0,
        exercise_min_week=150.0,
        sedentary_min_day=360.0,
        sleep_hours=7.5,
        phq9_score=3.0,
        phq9_category="minimal",
    )


def _minimal_input() -> PredictionInput:
    """Return a PredictionInput with only required fields (age, sex)."""
    return PredictionInput(age=55, sex=2)


def _response() -> PredictionResponse:
    return PredictionResponse(
        probability=0.42,
        risk_score=42.0,
        risk_band="Moderate",
        model_version="2",
        disclaimer="For educational purposes only.",
    )


def _fresh_logger(tmp_path: Path):
    """Create a fresh PredictionLogger pointing at tmp_path.

    Resets the singleton so each test is fully isolated.
    """
    from src.api.monitoring import PredictionLogger

    # Reset singleton state so __init__ runs again.
    PredictionLogger._instance = None
    with patch.dict(os.environ, {"PREDICTION_LOG_PATH": str(tmp_path / "logs" / "predictions.jsonl")}):
        logger = PredictionLogger()
    return logger


# ---------------------------------------------------------------------------
# 1. Nested "inputs" containing all 21 feature keys
# ---------------------------------------------------------------------------


class TestNestedInputs:
    """The JSONL entry must contain an ``"inputs"`` object with all 21 features."""

    def test_inputs_key_contains_all_21_features(self, tmp_path):
        logger = _fresh_logger(tmp_path)
        inp = _full_input()
        resp = _response()

        logger.log_prediction(inp, resp, request_id="req-001", latency_ms=12.345)

        line = logger.log_path.read_text().strip()
        entry = json.loads(line)

        assert "inputs" in entry, "Top-level 'inputs' key is missing"
        assert isinstance(entry["inputs"], dict)
        assert set(entry["inputs"].keys()) == set(ALL_21_FEATURES)
        assert len(entry["inputs"]) == 21

    def test_features_are_not_at_top_level(self, tmp_path):
        """Ensure features are nested inside 'inputs', not flattened."""
        logger = _fresh_logger(tmp_path)
        logger.log_prediction(_full_input(), _response(), "req-002", 10.0)

        entry = json.loads(logger.log_path.read_text().strip())
        for feat in ALL_21_FEATURES:
            assert feat not in entry, f"Feature '{feat}' must not be a top-level key"


# ---------------------------------------------------------------------------
# 2. Missing optional values serialised as JSON null
# ---------------------------------------------------------------------------


class TestNullSerialization:
    """Optional fields that are ``None`` must appear in the JSONL as ``null``."""

    def test_missing_optionals_are_null(self, tmp_path):
        logger = _fresh_logger(tmp_path)
        inp = _minimal_input()  # only age & sex provided
        resp = _response()

        logger.log_prediction(inp, resp, request_id="req-null", latency_ms=5.0)

        entry = json.loads(logger.log_path.read_text().strip())
        inputs = entry["inputs"]

        # Required fields present and non-null.
        assert inputs["age"] == 55
        assert inputs["sex"] == 2

        # All 19 optional fields must be explicitly null, not absent.
        optional_keys = [k for k in ALL_21_FEATURES if k not in ("age", "sex")]
        for key in optional_keys:
            assert key in inputs, f"Optional key '{key}' missing from inputs"
            assert inputs[key] is None, f"Optional key '{key}' should be null, got {inputs[key]!r}"


# ---------------------------------------------------------------------------
# 3. Exact JSONL contract
# ---------------------------------------------------------------------------


class TestJSONLContract:
    """Verify the top-level structure matches the Phase 3 handbook contract."""

    EXPECTED_TOP_KEYS = {
        "timestamp",
        "request_id",
        "model_version",
        "latency_ms",
        "inputs",
        "probability",
        "risk_score",
        "risk_band",
    }

    def test_top_level_keys_match_contract(self, tmp_path):
        logger = _fresh_logger(tmp_path)
        logger.log_prediction(_full_input(), _response(), "req-contract", 7.0)

        entry = json.loads(logger.log_path.read_text().strip())
        assert set(entry.keys()) == self.EXPECTED_TOP_KEYS

    def test_timestamp_is_utc_aware_iso8601(self, tmp_path):
        logger = _fresh_logger(tmp_path)
        logger.log_prediction(_full_input(), _response(), "req-tz", 1.0)

        entry = json.loads(logger.log_path.read_text().strip())
        ts = entry["timestamp"]
        # Must parse as a timezone-aware datetime with UTC offset.
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None, "Timestamp must be timezone-aware"
        assert parsed.utcoffset().total_seconds() == 0, "Timestamp must be UTC"

    def test_prediction_fields_match_response(self, tmp_path):
        logger = _fresh_logger(tmp_path)
        resp = _response()
        logger.log_prediction(_full_input(), resp, "req-pred", 3.5)

        entry = json.loads(logger.log_path.read_text().strip())
        assert entry["probability"] == resp.probability
        assert entry["risk_score"] == resp.risk_score
        assert entry["risk_band"] == resp.risk_band
        assert entry["model_version"] == resp.model_version

    def test_latency_ms_rounded_to_3_decimals(self, tmp_path):
        logger = _fresh_logger(tmp_path)
        logger.log_prediction(_full_input(), _response(), "req-lat", 12.34567)

        entry = json.loads(logger.log_path.read_text().strip())
        assert entry["latency_ms"] == 12.346

    def test_multiple_lines_produce_valid_jsonl(self, tmp_path):
        logger = _fresh_logger(tmp_path)
        for i in range(3):
            logger.log_prediction(_full_input(), _response(), f"req-{i}", float(i))

        lines = logger.log_path.read_text().strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            entry = json.loads(line)
            assert "inputs" in entry


# ---------------------------------------------------------------------------
# 4. Directory creation / file-opening failure (lazy init)
# ---------------------------------------------------------------------------


class TestLazyInitFailure:
    """File/directory errors during lazy init must warn, not crash."""

    def test_directory_creation_failure_logs_warning(self, tmp_path, caplog):
        logger = _fresh_logger(tmp_path)
        # Make _ensure_file raise by forcing mkdir to fail.
        with patch.object(Path, "mkdir", side_effect=PermissionError("mock denied")):
            with caplog.at_level(logging.WARNING, logger="health_api"):
                logger.log_prediction(_full_input(), _response(), "req-fail-dir", 1.0)

        assert any("Failed to write prediction log" in r.message for r in caplog.records)

    def test_file_open_failure_logs_warning(self, tmp_path, caplog):
        logger = _fresh_logger(tmp_path)
        # Directory creation succeeds but file open raises.
        with patch.object(Path, "open", side_effect=PermissionError("cannot open")):
            with caplog.at_level(logging.WARNING, logger="health_api"):
                logger.log_prediction(_full_input(), _response(), "req-fail-open", 1.0)

        assert any("Failed to write prediction log" in r.message for r in caplog.records)

    def test_init_does_not_touch_filesystem(self, tmp_path):
        """__init__ must not create directories or open files."""
        from src.api.monitoring import PredictionLogger

        PredictionLogger._instance = None
        fake_path = tmp_path / "nonexistent_dir" / "log.jsonl"
        with patch.dict(os.environ, {"PREDICTION_LOG_PATH": str(fake_path)}):
            logger = PredictionLogger()

        # The parent directory should NOT exist — init is lazy.
        assert not fake_path.parent.exists()
        assert logger._file is None


# ---------------------------------------------------------------------------
# 5. Write failure without breaking prediction logging integration
# ---------------------------------------------------------------------------


class TestWriteFailureSafety:
    """A write error must never prevent a prediction from succeeding."""

    def test_write_error_does_not_raise(self, tmp_path, caplog):
        logger = _fresh_logger(tmp_path)
        # Open file once so _ensure_file succeeds, then break writes.
        logger.log_prediction(_full_input(), _response(), "req-setup", 1.0)

        logger._file.write = MagicMock(side_effect=OSError("disk full"))
        with caplog.at_level(logging.WARNING, logger="health_api"):
            # Must not raise — just log a warning.
            logger.log_prediction(_full_input(), _response(), "req-broken-write", 2.0)

        assert any("Failed to write prediction log" in r.message for r in caplog.records)

    def test_subsequent_log_works_after_write_failure(self, tmp_path, caplog):
        """After a transient write failure, next log should still attempt to write."""
        logger = _fresh_logger(tmp_path)
        # First write succeeds.
        logger.log_prediction(_full_input(), _response(), "req-ok-1", 1.0)

        # Simulate a transient failure.
        original_write = logger._file.write
        logger._file.write = MagicMock(side_effect=OSError("transient"))
        with caplog.at_level(logging.WARNING, logger="health_api"):
            logger.log_prediction(_full_input(), _response(), "req-fail", 2.0)

        # Restore write.
        logger._file.write = original_write
        logger.log_prediction(_full_input(), _response(), "req-ok-2", 3.0)

        lines = [
            l for l in logger.log_path.read_text().strip().splitlines() if l.strip()
        ]
        # Should have 2 successful lines (first + third).
        assert len(lines) == 2
        ids = [json.loads(l)["request_id"] for l in lines]
        assert ids == ["req-ok-1", "req-ok-2"]

    def test_logger_is_singleton(self, tmp_path):
        """PredictionLogger should return the same instance."""
        from src.api.monitoring import PredictionLogger

        PredictionLogger._instance = None
        with patch.dict(os.environ, {"PREDICTION_LOG_PATH": str(tmp_path / "log.jsonl")}):
            a = PredictionLogger()
            b = PredictionLogger()
        assert a is b
