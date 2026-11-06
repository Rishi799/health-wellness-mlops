"""Public-service regressions using only a locally trained synthetic artifact."""

import json

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import skops.io as sio

from src.api import main as api
from src.api.predictor import Predictor
from src.api.schemas import PredictionInput


@pytest.fixture(scope="module")
def synthetic_artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-champion")
    features = list(PredictionInput.model_fields)
    categorical = ["sex", "education", "smoking_status", "phq9_category"]
    numeric = [f for f in features if f not in categorical]
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({f: rng.uniform(1, 10, 60) for f in numeric})
    frame["sex"] = np.tile([1, 2], 30)
    frame["education"] = np.tile([1., 2., 3., 4., 5.], 12)
    frame["smoking_status"] = np.tile(["never", "former", "current"], 20)
    frame["phq9_category"] = np.tile(["minimal", "mild", "moderate"], 20)
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([("preprocessing", prep), ("classifier", LogisticRegression())])
    model.fit(frame[features], np.tile([0, 1], 30))
    sio.dump(model, root / "model.skops")
    info = {"model_name": "synthetic-only", "model_version": "fixture",
            "features": features, "decision_threshold": 0.5, "metrics": {}}
    (root / "model_info.json").write_text(json.dumps(info), encoding="utf-8")
    return root


@pytest.fixture
def service(synthetic_artifact, monkeypatch):
    predictor = Predictor(synthetic_artifact / "model.skops", synthetic_artifact / "model_info.json")
    monkeypatch.setattr(api, "predictor", predictor)
    with TestClient(api.app) as client:
        yield client, predictor


@pytest.mark.parametrize("field,value", [
    ("education", 2.5), ("told_high_bp", 0.5),
    ("sex", True), ("age", True), ("education", True), ("sbp", False),
])
def test_invalid_encodings_rejected(service, field, value):
    client, _ = service
    response = client.post("/predict", json={"age": 55, "sex": 1, field: value})
    assert response.status_code == 422
    assert any(error["loc"][-1] == field for error in response.json()["detail"])


def test_valid_numeric_categories_and_missing_values_work(service):
    client, _ = service
    response = client.post("/predict", json={"age": 55, "sex": 1, "education": 3.0, "told_high_bp": 1.0})
    assert response.status_code == 200
    assert 0 <= response.json()["probability"] <= 1
    assert response.json()["disclaimer"]


def test_prediction_error_does_not_disclose_exception(service, monkeypatch):
    client, predictor = service
    def broken(_):
        raise RuntimeError("private/model/location INTERNAL_SENTINEL")
    monkeypatch.setattr(predictor, "predict", broken)
    response = client.post("/predict", json={"age": 55, "sex": 1})
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction could not be completed"}
    assert "INTERNAL_SENTINEL" not in response.text


def test_unavailable_endpoints_do_not_disclose_load_error(service):
    client, predictor = service
    predictor.is_loaded = False
    predictor.load_error = "INTERNAL_SENTINEL /private/model.skops"
    for method, path in [("get", "/health"), ("get", "/model-info"), ("post", "/predict")]:
        kwargs = {"json": {"age": 55, "sex": 1}} if method == "post" else {}
        response = getattr(client, method)(path, **kwargs)
        assert response.status_code == 503
        assert "INTERNAL_SENTINEL" not in response.text


def test_service_commit_is_distinct_from_training_commit(service, monkeypatch):
    client, _ = service
    monkeypatch.setenv("SERVICE_GIT_COMMIT", "a" * 40)
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["service_git_commit"] == "a" * 40


def test_startup_rejects_mismatched_metadata(synthetic_artifact, tmp_path):
    info = json.loads((synthetic_artifact / "model_info.json").read_text())
    info["features"].reverse()
    path = tmp_path / "bad-info.json"
    path.write_text(json.dumps(info))
    predictor = Predictor(synthetic_artifact / "model.skops", path)
    with pytest.raises(ValueError, match="feature ordering"):
        predictor.load()
    assert not predictor.is_loaded


def test_startup_rejects_model_that_cannot_predict(synthetic_artifact, monkeypatch):
    predictor = Predictor(synthetic_artifact / "model.skops", synthetic_artifact / "model_info.json")
    monkeypatch.setattr(Predictor, "predict", lambda *_: (_ for _ in ()).throw(ValueError("unusable estimator")))
    with pytest.raises(ValueError, match="unusable estimator"):
        predictor.load()
    assert not predictor.is_loaded


def test_invalid_probability_matrix_fails_closed(service, monkeypatch):
    client, predictor = service
    monkeypatch.setattr(predictor.model, "predict_proba", lambda _: np.array([[np.nan, np.nan]]))
    assert client.post("/predict", json={"age": 55, "sex": 1}).status_code == 500


def test_api_probability_matches_offline_at_documented_precision(service):
    client, predictor = service
    payload = PredictionInput(age=55, sex=1)
    frame = pd.DataFrame([{k: np.nan if v is None else v for k, v in payload.model_dump().items()}])
    expected = float(predictor.model.predict_proba(frame[predictor.features])[0, 1])
    result = client.post("/predict", json=payload.model_dump()).json()
    assert result["probability"] == round(expected, 4)
