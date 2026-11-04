"""Comprehensive unit and integration tests for FastAPI service (Phase 2 Task D).

Covers:
- Lifespan model loading and trusted-type whitelist security
- GET /health (HTTP 200 healthy, HTTP 503 when model fails to load)
- GET /model-info (HTTP 200 with 21-feature contract, HTTP 503 when unloaded)
- POST /predict:
  - Low-risk, moderate-risk, and high-risk patients
  - Missing optional values (graceful imputer handling)
  - Boundary value tests: exact 0.29, 0.30, 0.59, 0.60 risk-band boundaries
  - Invalid inputs: age out of range (20–80), invalid sex, invalid categorical values,
    out-of-bounds physiological values, forbidden extra attributes (HTTP 422)
  - Unloaded model rejection (HTTP 503)
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.predictor import (
    CLINICAL_DISCLAIMER,
    calculate_risk_band,
    predictor,
)


@pytest.fixture(autouse=True)
def ensure_predictor_loaded():
    """Ensure predictor is in a clean loaded state before each test."""
    predictor.load()
    yield
    # Restore standard state after tests
    predictor.load()


@pytest.fixture
def client():
    """Create FastAPI test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def low_risk_payload():
    """Realistic low-risk patient (young, healthy biometrics, active, never smoker)."""
    return {
        "age": 25,
        "sex": 2,
        "education": 4.0,
        "income_ratio": 4.5,
        "height_cm": 165.0,
        "weight_kg": 58.0,
        "bmi": 21.3,
        "waist_cm": 72.0,
        "sbp": 110.0,
        "dbp": 70.0,
        "pulse": 68.0,
        "told_high_bp": 0.0,
        "total_chol": 170.0,
        "hdl": 65.0,
        "smoking_status": "never",
        "alcohol_drinks_week": 0.0,
        "exercise_min_week": 240.0,
        "sedentary_min_day": 240.0,
        "sleep_hours": 8.0,
        "phq9_score": 1.0,
        "phq9_category": "minimal",
    }


@pytest.fixture
def moderate_risk_payload():
    """Realistic moderate-risk patient (middle-aged, prehypertension, moderate activity)."""
    return {
        "age": 52,
        "sex": 1,
        "education": 3.0,
        "income_ratio": 2.5,
        "height_cm": 172.0,
        "weight_kg": 82.0,
        "bmi": 27.7,
        "waist_cm": 92.0,
        "sbp": 128.0,
        "dbp": 82.0,
        "pulse": 74.0,
        "told_high_bp": 0.0,
        "total_chol": 210.0,
        "hdl": 45.0,
        "smoking_status": "former",
        "alcohol_drinks_week": 4.0,
        "exercise_min_week": 90.0,
        "sedentary_min_day": 360.0,
        "sleep_hours": 7.0,
        "phq9_score": 6.0,
        "phq9_category": "mild",
    }


@pytest.fixture
def high_risk_payload():
    """Realistic high-risk patient (older adult, stage 2 hypertension, obese, smoker)."""
    return {
        "age": 68,
        "sex": 1,
        "education": 2.0,
        "income_ratio": 1.2,
        "height_cm": 175.0,
        "weight_kg": 105.0,
        "bmi": 34.3,
        "waist_cm": 112.0,
        "sbp": 155.0,
        "dbp": 95.0,
        "pulse": 85.0,
        "told_high_bp": 1.0,
        "total_chol": 260.0,
        "hdl": 35.0,
        "smoking_status": "current",
        "alcohol_drinks_week": 14.0,
        "exercise_min_week": 0.0,
        "sedentary_min_day": 600.0,
        "sleep_hours": 5.0,
        "phq9_score": 16.0,
        "phq9_category": "moderately severe",
    }


# ==============================================================================
# 1. Deterministic Risk Band Boundary Tests
# ==============================================================================


@pytest.mark.parametrize(
    "score,expected_band",
    [
        (0.0, "Low"),
        (28.9, "Low"),
        (29.0, "Low"),
        (29.9, "Low"),
        (30.0, "Moderate"),
        (45.0, "Moderate"),
        (59.0, "Moderate"),
        (59.9, "Moderate"),
        (60.0, "High"),
        (60.1, "High"),
        (87.5, "High"),
        (100.0, "High"),
    ],
)
def test_calculate_risk_band_deterministic_scores(score, expected_band):
    """Test risk band categorization across all exact critical thresholds."""
    assert calculate_risk_band(score) == expected_band


@pytest.mark.parametrize(
    "prob,expected_score,expected_band",
    [
        (0.29, 29.0, "Low"),
        (0.30, 30.0, "Moderate"),
        (0.59, 59.0, "Moderate"),
        (0.60, 60.0, "High"),
    ],
)
def test_predict_endpoint_exact_boundary_probabilities(
    client, low_risk_payload, prob, expected_score, expected_band
):
    """Requirement 3: Deterministic test for risk-band boundaries: 0.29, 0.30, 0.59, 0.60."""
    with patch.object(
        predictor.model, "predict_proba", return_value=np.array([[1.0 - prob, prob]])
    ):
        response = client.post("/predict", json=low_risk_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["probability"] == prob
        assert data["risk_score"] == expected_score
        assert data["risk_band"] == expected_band


# ==============================================================================
# 2. GET /health Tests
# ==============================================================================


def test_health_endpoint_healthy(client):
    """Verify /health returns HTTP 200 when model is properly loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] == "2"
    assert data["error"] is None


def test_health_endpoint_503_when_model_fails_to_load(client):
    """Requirement 8: Return HTTP 503 from /health if the model fails to load."""
    predictor.is_loaded = False
    predictor.load_error = "Simulated disk failure or missing artifact"
    try:
        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False
        assert "Simulated disk failure" in data["error"]
    finally:
        predictor.load()


# ==============================================================================
# 3. GET /model-info Tests
# ==============================================================================


def test_model_info_endpoint_success(client):
    """Verify /model-info returns metadata matching models/model_info.json."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "cardiometabolic-risk-model"
    assert data["model_version"] == "2"
    assert len(data["features"]) == 21
    assert data["features"][0] == "age"
    assert data["features"][-1] == "phq9_category"
    assert data["decision_threshold"] == 0.5
    assert "test_roc_auc" in data
    assert "test_recall" in data


def test_model_info_503_when_unloaded(client):
    """Verify /model-info returns 503 when model is not loaded."""
    predictor.is_loaded = False
    predictor.load_error = "Metadata unavailable"
    try:
        response = client.get("/model-info")
        assert response.status_code == 503
    finally:
        predictor.load()


# ==============================================================================
# 4. POST /predict Clinical Inferences
# ==============================================================================


def test_predict_low_risk(client, low_risk_payload):
    """Verify end-to-end low-risk prediction."""
    response = client.post("/predict", json=low_risk_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_band"] == "Low"
    assert data["risk_score"] < 30.0
    assert 0.0 <= data["probability"] <= 1.0
    assert data["model_version"] == "2"
    assert data["disclaimer"] == CLINICAL_DISCLAIMER


def test_predict_high_risk(client, high_risk_payload):
    """Verify end-to-end high-risk prediction."""
    response = client.post("/predict", json=high_risk_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_band"] == "High"
    assert data["risk_score"] >= 60.0
    assert 0.0 <= data["probability"] <= 1.0
    assert data["model_version"] == "2"
    assert data["disclaimer"] == CLINICAL_DISCLAIMER


def test_predict_moderate_risk(client, moderate_risk_payload):
    """Verify end-to-end moderate-risk prediction."""
    response = client.post("/predict", json=moderate_risk_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_band"] == "Moderate"
    assert 30.0 <= data["risk_score"] < 60.0


def test_predict_missing_values_allowed(client):
    """Requirement 5 & 9: Support optional/missing input values where allowed."""
    minimal_payload = {
        "age": 45,
        "sex": 1,
        # All 19 optional features omitted -> handled by pipeline imputers
    }
    response = client.post("/predict", json=minimal_payload)
    assert response.status_code == 200
    data = response.json()
    assert "probability" in data
    assert "risk_score" in data
    assert data["risk_band"] in ["Low", "Moderate", "High"]
    assert data["model_version"] == "2"


def test_predict_explicit_nulls(client):
    """Verify explicit None / null values in optional fields are handled cleanly."""
    nulls_payload = {
        "age": 55,
        "sex": 2,
        "education": None,
        "income_ratio": None,
        "height_cm": None,
        "weight_kg": None,
        "bmi": None,
        "waist_cm": None,
        "sbp": None,
        "dbp": None,
        "pulse": None,
        "told_high_bp": None,
        "total_chol": None,
        "hdl": None,
        "smoking_status": None,
        "alcohol_drinks_week": None,
        "exercise_min_week": None,
        "sedentary_min_day": None,
        "sleep_hours": None,
        "phq9_score": None,
        "phq9_category": None,
    }
    response = client.post("/predict", json=nulls_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_band"] in ["Low", "Moderate", "High"]


# ==============================================================================
# 5. POST /predict Validation Errors (HTTP 422)
# ==============================================================================


@pytest.mark.parametrize(
    "invalid_patch,field_name",
    [
        # Age validation: adult cohort 20–80 per validate.py
        ({"age": 19}, "age"),
        ({"age": 81}, "age"),
        ({"age": 10}, "age"),
        ({"age": 150}, "age"),
        ({"age": -5}, "age"),
        # Sex: 1=Male, 2=Female
        ({"sex": 0}, "sex"),
        ({"sex": 3}, "sex"),
        # Education: 1.0–5.0
        ({"education": 0.5}, "education"),
        ({"education": 6.0}, "education"),
        # Income ratio: 0.0–5.0
        ({"income_ratio": -0.1}, "income_ratio"),
        ({"income_ratio": 5.5}, "income_ratio"),
        # Height: 50.0–230.0
        ({"height_cm": 45.0}, "height_cm"),
        ({"height_cm": 235.0}, "height_cm"),
        # Weight: 2.0–300.0
        ({"weight_kg": 1.0}, "weight_kg"),
        ({"weight_kg": 350.0}, "weight_kg"),
        # BMI: 12.0–80.0
        ({"bmi": 10.0}, "bmi"),
        ({"bmi": 85.0}, "bmi"),
        # Waist: 30.0–200.0
        ({"waist_cm": 25.0}, "waist_cm"),
        ({"waist_cm": 210.0}, "waist_cm"),
        # SBP: 60.0–260.0
        ({"sbp": 55.0}, "sbp"),
        ({"sbp": 270.0}, "sbp"),
        # DBP: 20.0–150.0
        ({"dbp": 15.0}, "dbp"),
        ({"dbp": 160.0}, "dbp"),
        # Pulse: 30.0–200.0
        ({"pulse": 25.0}, "pulse"),
        ({"pulse": 210.0}, "pulse"),
        # Told high BP: binary 0.0 or 1.0
        ({"told_high_bp": 2.0}, "told_high_bp"),
        ({"told_high_bp": -1.0}, "told_high_bp"),
        # Total cholesterol: 50.0–600.0
        ({"total_chol": 40.0}, "total_chol"),
        ({"total_chol": 650.0}, "total_chol"),
        # HDL: 10.0–200.0
        ({"hdl": 5.0}, "hdl"),
        ({"hdl": 210.0}, "hdl"),
        # Smoking status: "never" | "former" | "current"
        ({"smoking_status": "vaping"}, "smoking_status"),
        ({"smoking_status": "occasional"}, "smoking_status"),
        # Alcohol: 0.0–110.0
        ({"alcohol_drinks_week": 115.0}, "alcohol_drinks_week"),
        # Exercise: 0.0–3000.0
        ({"exercise_min_week": 3500.0}, "exercise_min_week"),
        # Sedentary: 0.0–1440.0
        ({"sedentary_min_day": 1500.0}, "sedentary_min_day"),
        # Sleep: 2.0–14.0
        ({"sleep_hours": 1.0}, "sleep_hours"),
        ({"sleep_hours": 16.0}, "sleep_hours"),
        # PHQ-9 score: 0.0–27.0
        ({"phq9_score": 28.0}, "phq9_score"),
        ({"phq9_score": -1.0}, "phq9_score"),
        # PHQ-9 category
        ({"phq9_category": "extreme"}, "phq9_category"),
        # Extra unknown key (forbidden)
        ({"unrecognized_extra_key": 123}, "unrecognized_extra_key"),
        # Wrong type
        ({"age": "forty-five"}, "age"),
    ],
)
def test_predict_validation_errors(client, low_risk_payload, invalid_patch, field_name):
    """Requirement 8: Return HTTP 422 for invalid input."""
    payload = dict(low_risk_payload)
    payload.update(invalid_patch)
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_503_when_unloaded(client, low_risk_payload):
    """Verify /predict returns 503 if model is not loaded."""
    predictor.is_loaded = False
    predictor.load_error = "Model not initialized"
    try:
        response = client.post("/predict", json=low_risk_payload)
        assert response.status_code == 503
    finally:
        predictor.load()


# ==============================================================================
# 6. Predictor Whitelist Security Policy
# ==============================================================================


def test_predictor_untrusted_type_security_rejection(tmp_path):
    """Requirement 2: Strict enforcement of approved skops whitelist."""
    from src.api.predictor import Predictor

    dummy_model_file = tmp_path / "corrupt_model.skops"
    dummy_model_file.write_text("dummy")
    dummy_info_file = tmp_path / "info.json"
    dummy_info_file.write_text('{"features": ["age"], "model_version": "1"}')

    test_pred = Predictor(model_path=dummy_model_file, info_path=dummy_info_file)

    with patch("skops.io.get_untrusted_types", return_value=["os.system", "builtins.eval"]):
        with pytest.raises(ValueError, match="Security policy violation"):
            test_pred.load()

    assert test_pred.is_loaded is False
    assert "Security policy violation" in test_pred.load_error

