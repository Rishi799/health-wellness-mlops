"""Offline payload, pacing and request accounting checks for batch replay."""

from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from scripts import simulate_traffic as traffic
from src.api.schemas import PredictionInput


def test_payload_has_only_serving_features_and_json_scalars():
    payload = traffic.row_to_payload(pd.Series({
        "seqn": 42, "cardiometabolic": 1, "hba1c": 6, "diet_quality": 3,
        "age": np.float64(55), "sex": np.int64(1), "education": 3,
        "told_high_bp": 1, "bmi": np.float64(30), "hdl": np.nan,
        "smoking_status": "former", "phq9_category": "minimal",
    }))
    assert set(payload) <= set(PredictionInput.model_fields)
    assert type(payload["age"]) is int
    assert type(payload["sex"]) is int
    assert type(payload["education"]) is float
    assert type(payload["told_high_bp"]) is float
    assert "hdl" not in payload
    assert payload["smoking_status"] == "former"
    PredictionInput.model_validate(payload)


def test_invalid_codes_are_preserved_instead_of_truncated():
    payload = traffic.row_to_payload({"age": 55.5, "sex": 1, "education": 2.5})
    assert payload["age"] == 55.5
    assert payload["education"] == 2.5
    with pytest.raises(ValueError):
        PredictionInput.model_validate(payload)


def test_empty_and_missing_values_are_omitted():
    assert traffic.row_to_payload({"bmi": np.nan, "hdl": None, "education": pd.NA}) == {}


def test_replay_counts_statuses_limits_rows_and_paces_bad_requests(monkeypatch):
    now = [0.0]
    starts = []
    payloads = []
    monkeypatch.setattr(traffic.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(traffic.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    statuses = iter([200, 422, 503])

    def handler(request):
        starts.append(now[0])
        payloads.append(request.content)
        return httpx.Response(next(statuses))

    frame = pd.DataFrame({"age": [55, 60, 70], "sex": [1, 2, 1]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        summary = traffic.replay_batch(frame, "http://localhost:8000", rate=2, limit=2,
                                       bad=1, client=client)
    assert starts == [0, 0.5, 1]
    assert summary == {"sent": 3, "200": 1, "422": 1, "other": 1,
                       "network_errors": 0, "mean_latency_ms": 0}
    assert b'"education":2.5' in payloads[-1]


def test_network_errors_are_counted_without_retry(monkeypatch):
    monkeypatch.setattr(traffic.time, "sleep", lambda _: None)

    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        summary = traffic.replay_batch(pd.DataFrame({"age": [55], "sex": [1]}),
                                       "http://localhost:8000", client=client)
    assert summary["network_errors"] == summary["sent"] == 1
    assert summary["other"] == 0


@pytest.mark.parametrize("kwargs", [{"rate": 0}, {"rate": float("nan")}, {"timeout": -1},
                                    {"limit": -1}, {"bad": -1}])
def test_invalid_controls_fail_before_any_request(kwargs):
    with pytest.raises(ValueError):
        traffic.replay_batch(pd.DataFrame(), "http://localhost:8000", **kwargs)


def test_remote_workload_is_bounded():
    with pytest.raises(ValueError, match="Remote demos"):
        traffic.replay_batch(pd.DataFrame({"age": [55] * 21}), "https://example.org", rate=1)
    with pytest.raises(ValueError, match="Remote demos"):
        traffic.replay_batch(pd.DataFrame(), "https://example.org", rate=5)


@pytest.mark.parametrize("url", ["ftp://example.org", "http://user:secret@localhost", "http://localhost/health"])
def test_invalid_url_is_rejected(url):
    with pytest.raises(ValueError):
        traffic.validate_api_url(url)


@pytest.mark.real_artifacts
def test_replay_against_fastapi(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    if not (root / "models/model.skops").exists():
        pytest.skip("requires DVC champion artifact")
    from fastapi.testclient import TestClient
    from src.api.main import app

    monkeypatch.setattr(traffic.time, "sleep", lambda _: None)
    with TestClient(app) as client:
        summary = traffic.replay_batch(pd.DataFrame({"age": [55], "sex": [1]}),
                                       "http://localhost:8000", bad=1, client=client)
    assert summary["200"] == summary["422"] == 1
    assert summary["sent"] == 2
    assert summary["other"] == summary["network_errors"] == 0
