"""Integration tests for Phase 3 Task 1 — Request-ID middleware and monitoring.

Covers:
1. Three successful predictions (low, moderate, high risk).
2. Invalid request returns 422 and does NOT trigger PredictionLogger.
3. X-Request-ID header is present in every response.
4. Client-supplied X-Request-ID is echoed back unchanged.
5. JSON response bodies are unchanged (no extra monitoring keys).
6. PredictionLogger failure never affects prediction response.
7. PredictionLogger receives all 21 input fields including null values.
8. Request logs contain method, path, status, latency_ms, and request_id.
9. Request logs never contain patient input data.
"""

import json
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, _prediction_logger
from src.api.predictor import predictor, CLINICAL_DISCLAIMER

pytestmark = pytest.mark.real_artifacts

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ALL_21_FEATURES = [
    "age", "sex", "education", "income_ratio", "height_cm", "weight_kg",
    "bmi", "waist_cm", "sbp", "dbp", "pulse", "told_high_bp", "total_chol",
    "hdl", "smoking_status", "alcohol_drinks_week", "exercise_min_week",
    "sedentary_min_day", "sleep_hours", "phq9_score", "phq9_category",
]

EXPECTED_PREDICT_KEYS = {"probability", "risk_score", "risk_band", "model_version", "disclaimer"}


@pytest.fixture(autouse=True)
def ensure_predictor_loaded():
    """Guarantee predictor is loaded before every test in this module."""
    predictor.load()
    yield
    predictor.load()


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def low_risk_payload():
    return {
        "age": 25, "sex": 2, "education": 4.0, "income_ratio": 4.5,
        "height_cm": 165.0, "weight_kg": 58.0, "bmi": 21.3, "waist_cm": 72.0,
        "sbp": 110.0, "dbp": 70.0, "pulse": 68.0, "told_high_bp": 0.0,
        "total_chol": 170.0, "hdl": 65.0, "smoking_status": "never",
        "alcohol_drinks_week": 0.0, "exercise_min_week": 240.0,
        "sedentary_min_day": 240.0, "sleep_hours": 8.0,
        "phq9_score": 1.0, "phq9_category": "minimal",
    }


@pytest.fixture
def moderate_risk_payload():
    return {
        "age": 52, "sex": 1, "education": 3.0, "income_ratio": 2.5,
        "height_cm": 172.0, "weight_kg": 82.0, "bmi": 27.7, "waist_cm": 92.0,
        "sbp": 128.0, "dbp": 82.0, "pulse": 74.0, "told_high_bp": 0.0,
        "total_chol": 210.0, "hdl": 45.0, "smoking_status": "former",
        "alcohol_drinks_week": 4.0, "exercise_min_week": 90.0,
        "sedentary_min_day": 360.0, "sleep_hours": 7.0,
        "phq9_score": 6.0, "phq9_category": "mild",
    }


@pytest.fixture
def high_risk_payload():
    return {
        "age": 68, "sex": 1, "education": 2.0, "income_ratio": 1.2,
        "height_cm": 175.0, "weight_kg": 105.0, "bmi": 34.3, "waist_cm": 112.0,
        "sbp": 155.0, "dbp": 95.0, "pulse": 85.0, "told_high_bp": 1.0,
        "total_chol": 260.0, "hdl": 35.0, "smoking_status": "current",
        "alcohol_drinks_week": 14.0, "exercise_min_week": 0.0,
        "sedentary_min_day": 600.0, "sleep_hours": 5.0,
        "phq9_score": 16.0, "phq9_category": "moderately severe",
    }


# ---------------------------------------------------------------------------
# 1. Three successful predictions
# ---------------------------------------------------------------------------


class TestSuccessfulPredictions:
    """POST /predict succeeds for low, moderate, and high-risk patients."""

    def test_low_risk_prediction(self, client, low_risk_payload):
        response = client.post("/predict", json=low_risk_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["risk_band"] == "Low"
        assert data["risk_score"] < 30.0
        assert 0.0 <= data["probability"] <= 1.0
        assert data["model_version"] == "2"
        assert data["disclaimer"] == CLINICAL_DISCLAIMER

    def test_moderate_risk_prediction(self, client, moderate_risk_payload):
        response = client.post("/predict", json=moderate_risk_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["risk_band"] == "Moderate"
        assert 30.0 <= data["risk_score"] < 60.0

    def test_high_risk_prediction(self, client, high_risk_payload):
        response = client.post("/predict", json=high_risk_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["risk_band"] == "High"
        assert data["risk_score"] >= 60.0


# ---------------------------------------------------------------------------
# 2. Invalid request → 422, logger NOT called
# ---------------------------------------------------------------------------


class TestInvalidRequestBehavior:
    """422 responses must never trigger PredictionLogger."""

    def test_invalid_age_returns_422(self, client, low_risk_payload):
        payload = dict(low_risk_payload)
        payload["age"] = 150  # out of range (20–80)
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_logger_not_called_on_422(self, client, low_risk_payload):
        payload = dict(low_risk_payload)
        payload["sex"] = 9  # invalid
        with patch.object(_prediction_logger, "log_prediction") as mock_log:
            response = client.post("/predict", json=payload)
        assert response.status_code == 422
        mock_log.assert_not_called()

    def test_invalid_smoking_status_returns_422(self, client, low_risk_payload):
        payload = dict(low_risk_payload)
        payload["smoking_status"] = "vaping"
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# 3. X-Request-ID header on every response
# ---------------------------------------------------------------------------


class TestRequestIDHeader:
    """Every response must carry an X-Request-ID header."""

    def test_predict_response_has_request_id(self, client, low_risk_payload):
        response = client.post("/predict", json=low_risk_payload)
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0

    def test_health_response_has_request_id(self, client):
        response = client.get("/health")
        assert "X-Request-ID" in response.headers

    def test_model_info_response_has_request_id(self, client):
        response = client.get("/model-info")
        assert "X-Request-ID" in response.headers

    def test_invalid_predict_response_has_request_id(self, client, low_risk_payload):
        payload = dict(low_risk_payload)
        payload["age"] = 999
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        assert "X-Request-ID" in response.headers

    def test_client_supplied_request_id_is_echoed(self, client, low_risk_payload):
        custom_id = "my-trace-id-abc123"
        response = client.post(
            "/predict",
            json=low_risk_payload,
            headers={"X-Request-ID": custom_id},
        )
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == custom_id

    def test_generated_request_id_is_uuid_format(self, client, low_risk_payload):
        """Auto-generated IDs must be valid UUID v4 strings."""
        import uuid
        response = client.post("/predict", json=low_risk_payload)
        rid = response.headers["X-Request-ID"]
        # Should not raise if it is a valid UUID.
        parsed = uuid.UUID(rid)
        assert parsed.version == 4

    def test_different_requests_get_different_ids(self, client, low_risk_payload):
        r1 = client.post("/predict", json=low_risk_payload)
        r2 = client.post("/predict", json=low_risk_payload)
        assert r1.headers["X-Request-ID"] != r2.headers["X-Request-ID"]


# ---------------------------------------------------------------------------
# 4. Unchanged response bodies
# ---------------------------------------------------------------------------


class TestResponseBodyUnchanged:
    """Monitoring middleware must not alter any existing response body fields."""

    def test_predict_body_has_exactly_the_expected_keys(self, client, low_risk_payload):
        response = client.post("/predict", json=low_risk_payload)
        assert response.status_code == 200
        assert set(response.json().keys()) == EXPECTED_PREDICT_KEYS

    def test_no_monitoring_keys_in_predict_body(self, client, low_risk_payload):
        response = client.post("/predict", json=low_risk_payload)
        body = response.json()
        for forbidden in ("request_id", "latency_ms", "timestamp", "inputs"):
            assert forbidden not in body, f"Unexpected monitoring key '{forbidden}' in response body"

    def test_health_body_keys_unchanged(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert set(response.json().keys()) == {"status", "model_loaded", "model_version", "error"}

    def test_model_info_body_contains_expected_fields(self, client):
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "features" in data
        assert len(data["features"]) == 21


# ---------------------------------------------------------------------------
# 5. PredictionLogger failure does not affect response
# ---------------------------------------------------------------------------


class TestLoggingFailureSafety:
    """Logging errors must be silently swallowed — responses must be unaffected."""

    def test_logger_exception_does_not_affect_200_response(self, client, low_risk_payload):
        with patch.object(
            _prediction_logger,
            "log_prediction",
            side_effect=RuntimeError("disk full"),
        ):
            response = client.post("/predict", json=low_risk_payload)
        assert response.status_code == 200
        data = response.json()
        assert "risk_band" in data
        assert "probability" in data

    def test_logger_oserror_does_not_affect_response(self, client, high_risk_payload):
        with patch.object(
            _prediction_logger,
            "log_prediction",
            side_effect=OSError("permission denied"),
        ):
            response = client.post("/predict", json=high_risk_payload)
        assert response.status_code == 200

    def test_response_body_intact_despite_logger_failure(self, client, moderate_risk_payload):
        with patch.object(
            _prediction_logger,
            "log_prediction",
            side_effect=Exception("unexpected"),
        ):
            response = client.post("/predict", json=moderate_risk_payload)
        assert response.status_code == 200
        assert set(response.json().keys()) == EXPECTED_PREDICT_KEYS


# ---------------------------------------------------------------------------
# 6. Logger receives all 21 input fields including null values
# ---------------------------------------------------------------------------


class TestLoggerReceivesAllFields:
    """PredictionLogger.log_prediction must receive all 21 features, including nulls."""

    def test_logger_called_with_all_21_fields_full_payload(self, client, low_risk_payload):
        captured = {}

        def capture(input_data, response, request_id, latency_ms):
            captured["inputs"] = input_data.model_dump()
            captured["request_id"] = request_id
            captured["latency_ms"] = latency_ms

        with patch.object(_prediction_logger, "log_prediction", side_effect=capture):
            resp = client.post("/predict", json=low_risk_payload)

        assert resp.status_code == 200
        assert set(captured["inputs"].keys()) == set(ALL_21_FEATURES)
        assert len(captured["inputs"]) == 21
        # Non-null fields must carry their values.
        assert captured["inputs"]["age"] == 25
        assert captured["inputs"]["sex"] == 2

    def test_logger_called_with_null_optionals(self, client):
        """When only age+sex are provided, all 19 optional fields must be passed as None."""
        minimal_payload = {"age": 45, "sex": 1}
        captured = {}

        def capture(input_data, response, request_id, latency_ms):
            captured["inputs"] = input_data.model_dump()

        with patch.object(_prediction_logger, "log_prediction", side_effect=capture):
            resp = client.post("/predict", json=minimal_payload)

        assert resp.status_code == 200
        inputs = captured["inputs"]
        assert set(inputs.keys()) == set(ALL_21_FEATURES)

        optional_keys = [k for k in ALL_21_FEATURES if k not in ("age", "sex")]
        for key in optional_keys:
            assert key in inputs, f"Optional key '{key}' missing"
            assert inputs[key] is None, f"Expected None for '{key}', got {inputs[key]!r}"

    def test_logger_called_with_explicit_nulls_in_payload(self, client):
        """Explicit null values in the JSON body must arrive as None in the logger."""
        nulls_payload = {
            "age": 55, "sex": 2,
            "education": None, "income_ratio": None, "height_cm": None,
            "weight_kg": None, "bmi": None, "waist_cm": None, "sbp": None,
            "dbp": None, "pulse": None, "told_high_bp": None,
            "total_chol": None, "hdl": None, "smoking_status": None,
            "alcohol_drinks_week": None, "exercise_min_week": None,
            "sedentary_min_day": None, "sleep_hours": None,
            "phq9_score": None, "phq9_category": None,
        }
        captured = {}

        def capture(input_data, response, request_id, latency_ms):
            captured["inputs"] = input_data.model_dump()

        with patch.object(_prediction_logger, "log_prediction", side_effect=capture):
            resp = client.post("/predict", json=nulls_payload)

        assert resp.status_code == 200
        inputs = captured["inputs"]
        for key in ALL_21_FEATURES:
            if key in ("age", "sex"):
                assert inputs[key] is not None
            else:
                assert inputs[key] is None, f"Expected null for '{key}'"

    def test_logger_receives_request_id_matching_response_header(self, client, low_risk_payload):
        """The request_id passed to PredictionLogger must match X-Request-ID in the response."""
        captured = {}

        def capture(input_data, response, request_id, latency_ms):
            captured["request_id"] = request_id

        with patch.object(_prediction_logger, "log_prediction", side_effect=capture):
            resp = client.post("/predict", json=low_risk_payload)

        assert resp.status_code == 200
        assert captured["request_id"] == resp.headers["X-Request-ID"]

    def test_logger_receives_positive_latency_ms(self, client, low_risk_payload):
        """latency_ms passed to PredictionLogger must be a positive float."""
        captured = {}

        def capture(input_data, response, request_id, latency_ms):
            captured["latency_ms"] = latency_ms

        with patch.object(_prediction_logger, "log_prediction", side_effect=capture):
            client.post("/predict", json=low_risk_payload)

        assert isinstance(captured["latency_ms"], float)
        assert captured["latency_ms"] > 0


# ---------------------------------------------------------------------------
# 7. X-Request-ID validation (allowlist + length cap)
# ---------------------------------------------------------------------------


class TestRequestIDValidation:
    """Client-supplied X-Request-ID must be validated; invalid values are replaced."""

    def test_valid_client_id_is_preserved(self, client, low_risk_payload):
        """A conforming client ID (A-Za-z0-9, hyphen, underscore, ≤128 chars) is echoed."""
        valid_id = "trace-id_ABC123-xyz"
        response = client.post(
            "/predict",
            json=low_risk_payload,
            headers={"X-Request-ID": valid_id},
        )
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == valid_id

    def test_id_longer_than_128_chars_is_replaced(self, client, low_risk_payload):
        """An ID of 129+ characters must be silently replaced with a fresh UUID v4."""
        import uuid as _uuid
        too_long = "a" * 129
        response = client.post(
            "/predict",
            json=low_risk_payload,
            headers={"X-Request-ID": too_long},
        )
        assert response.status_code == 200
        returned_id = response.headers["X-Request-ID"]
        assert returned_id != too_long
        # Replacement must be a valid UUID v4.
        parsed = _uuid.UUID(returned_id)
        assert parsed.version == 4

    def test_id_with_invalid_characters_is_replaced(self, client, low_risk_payload):
        """An ID containing characters outside [A-Za-z0-9\\-_] is replaced with UUID v4."""
        import uuid as _uuid
        invalid_ids = [
            "has space",
            "has@at",
            "has/slash",
            "has.dot",
            "has<script>",
        ]
        for bad_id in invalid_ids:
            response = client.post(
                "/predict",
                json=low_risk_payload,
                headers={"X-Request-ID": bad_id},
            )
            assert response.status_code == 200
            returned_id = response.headers["X-Request-ID"]
            assert returned_id != bad_id, f"Expected replacement for invalid ID: {bad_id!r}"
            parsed = _uuid.UUID(returned_id)
            assert parsed.version == 4

    def test_missing_id_generates_uuid_v4(self, client, low_risk_payload):
        """When X-Request-ID header is absent entirely, a UUID v4 is generated."""
        import uuid as _uuid
        response = client.post("/predict", json=low_risk_payload)
        assert response.status_code == 200
        returned_id = response.headers["X-Request-ID"]
        parsed = _uuid.UUID(returned_id)
        assert parsed.version == 4

    def test_exactly_128_char_id_is_preserved(self, client, low_risk_payload):
        """An ID of exactly 128 valid characters must be accepted (boundary check)."""
        boundary_id = "A" * 128
        response = client.post(
            "/predict",
            json=low_risk_payload,
            headers={"X-Request-ID": boundary_id},
        )
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == boundary_id


# ---------------------------------------------------------------------------
# 8. Request logs contain required fields and never patient data
# ---------------------------------------------------------------------------


class TestRequestLogContent:
    """The health_api logger must emit structured info with specific fields."""

    def test_request_log_contains_required_fields(self, client, low_risk_payload, caplog):
        """Each request log record must include method, path, status, latency_ms, request_id."""
        with caplog.at_level(logging.INFO, logger="health_api"):
            response = client.post("/predict", json=low_risk_payload)

        assert response.status_code == 200

        # Find the middleware "request completed" record.
        req_records = [
            r for r in caplog.records
            if r.name == "health_api" and r.message == "request completed"
        ]
        assert len(req_records) >= 1, "No 'request completed' log record found"
        rec = req_records[-1]

        # All five structured fields must be present in the log record's extras.
        assert rec.method == "POST"
        assert rec.path == "/predict"
        assert rec.status == 200
        assert isinstance(rec.latency_ms, (int, float)) and rec.latency_ms > 0
        assert hasattr(rec, "request_id") and len(rec.request_id) > 0

    def test_request_log_never_contains_patient_data(self, client, low_risk_payload, caplog):
        """No log record from health_api should contain patient input values."""
        with caplog.at_level(logging.DEBUG, logger="health_api"):
            client.post("/predict", json=low_risk_payload)

        # Sensitive field values that must NOT appear in any log message.
        sensitive_markers = [
            '"age"', '"bmi"', '"waist_cm"', '"sbp"', '"hdl"',
            '"smoking_status"', '"phq9_score"', '"phq9_category"',
        ]
        for rec in caplog.records:
            if rec.name != "health_api":
                continue
            msg = rec.getMessage()
            for marker in sensitive_markers:
                assert marker not in msg, (
                    f"Patient data key {marker} leaked into log message: {msg!r}"
                )
