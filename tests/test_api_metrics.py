"""Offline-compatible unit and integration tests for the six Prometheus API metrics."""

import json
import logging
import math
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import skops.io as sio
from fastapi.testclient import TestClient
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, CollectorRegistry
from prometheus_client.parser import text_string_to_metric_families
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.api import main as api
from src.api.monitoring import APIMetrics, create_metrics_registry
from src.api.predictor import Predictor
from src.api.schemas import PredictionInput, PredictionResponse


EXPECTED_METRIC_TYPES = {
    "predictions_total": "counter",
    "prediction_risk_score": "histogram",
    "prediction_probability": "histogram",
    "model_info": "gauge",
    "api_requests_total": "counter",
    "api_request_latency_seconds": "histogram",
}

EXPECTED_PARSED_FAMILY_TYPES = {
    "predictions": "counter",
    "prediction_risk_score": "histogram",
    "prediction_probability": "histogram",
    "model_info": "gauge",
    "api_requests": "counter",
    "api_request_latency_seconds": "histogram",
}

EXPECTED_METRIC_LABELS = {
    "predictions_total": ("risk_band",),
    "prediction_risk_score": (),
    "prediction_probability": (),
    "model_info": ("model_name", "model_version", "git_commit"),
    "api_requests_total": ("endpoint", "method", "status"),
    "api_request_latency_seconds": ("endpoint",),
}

EXPECTED_RISK_SCORE_BOUNDS = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, math.inf]
EXPECTED_RISK_SCORE_LE = ["0.0", "10.0", "20.0", "30.0", "40.0", "50.0", "60.0", "70.0", "80.0", "90.0", "100.0", "+Inf"]

EXPECTED_PROBABILITY_BOUNDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, math.inf]
EXPECTED_PROBABILITY_LE = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "+Inf"]

FORBIDDEN_EXTRA_METRICS = {"prediction_inference_latency_seconds", "model_loaded"}


# ---------------------------------------------------------------------------
# Helpers & Offline Synthetic Fixture
# ---------------------------------------------------------------------------


def _parse_metrics(text: str) -> dict[tuple[str, frozenset[tuple[str, str]]], float]:
    """Parse Prometheus exposition text into {(sample_name, frozenset(labels)): value}."""
    parsed: dict[tuple[str, frozenset[tuple[str, str]]], float] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            parsed[(sample.name, frozenset(sample.labels.items()))] = sample.value
    return parsed


def _sample_value(
    parsed: dict[tuple[str, frozenset[tuple[str, str]]], float],
    name: str,
    **labels: str,
) -> float:
    """Return a metric sample value or 0.0 if the labelset has not been emitted yet."""
    return parsed.get((name, frozenset(labels.items())), 0.0)


def _sum_samples(
    parsed: dict[tuple[str, frozenset[tuple[str, str]]], float],
    name: str,
) -> float:
    """Sum all sample values for a metric name across every label combination."""
    return sum(val for (sample_name, _), val in parsed.items() if sample_name == name)


def _all_label_pairs(
    parsed: dict[tuple[str, frozenset[tuple[str, str]]], float],
    name: str,
) -> list[dict[str, str]]:
    """Return all label dictionaries emitted for a given sample name."""
    return [dict(labels) for (sample_name, labels) in parsed if sample_name == name]


def _app_families(text: str) -> dict[str, object]:
    """Return non-runtime, non-*_created parsed metric families keyed by family name."""
    return {
        family.name: family
        for family in text_string_to_metric_families(text)
        if not family.name.startswith(("process_", "python_"))
        and not family.name.endswith("_created")
    }


def _app_declared_types(text: str) -> dict[str, str]:
    """Return {# TYPE name: type} declarations excluding runtime and *_created gauges."""
    declared: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("# TYPE "):
            _, name, metric_type = line.split(None, 3)[1:]
            if not name.startswith(("process_", "python_")) and not name.endswith("_created"):
                declared[name] = metric_type
    return declared


def _bucket_les(family) -> list[str]:
    """Extract ordered unique 'le' bucket boundaries from a histogram family."""
    seen: list[str] = []
    for sample in family.samples:
        if sample.name == f"{family.name}_bucket":
            le = sample.labels["le"]
            if le not in seen:
                seen.append(le)
    return seen


@pytest.fixture(scope="module")
def synthetic_artifact(tmp_path_factory):
    """Train a tiny local skops model matching the 21-feature serving contract."""
    root = tmp_path_factory.mktemp("synthetic-metrics-model")
    features = list(PredictionInput.model_fields)
    categorical = ["sex", "education", "smoking_status", "phq9_category"]
    numeric = [f for f in features if f not in categorical]

    rng = np.random.default_rng(7)
    frame = pd.DataFrame({f: rng.uniform(1, 10, 60) for f in numeric})
    frame["sex"] = np.tile([1, 2], 30)
    frame["education"] = np.tile([1.0, 2.0, 3.0, 4.0, 5.0], 12)
    frame["smoking_status"] = np.tile(["never", "former", "current"], 20)
    frame["phq9_category"] = np.tile(["minimal", "mild", "moderate"], 20)

    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    model = Pipeline([("preprocessing", prep), ("classifier", LogisticRegression())])
    model.fit(frame[features], np.tile([0, 1], 30))

    sio.dump(model, root / "model.skops")
    info = {
        "model_name": "synthetic-cardiometabolic-model",
        "model_version": "99",
        "git_commit": "training-commit-should-not-be-used-in-gauge",
        "features": features,
        "decision_threshold": 0.5,
        "metrics": {"test_roc_auc": 0.85, "test_recall": 0.80},
    }
    (root / "model_info.json").write_text(json.dumps(info), encoding="utf-8")
    return root


@pytest.fixture
def isolated_log_path(tmp_path, monkeypatch):
    """Point PredictionLogger at a temporary path for offline tests."""
    log_path = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(api._prediction_logger, "log_path", log_path)
    monkeypatch.setattr(api._prediction_logger, "_file", None)
    yield log_path
    fh = getattr(api._prediction_logger, "_file", None)
    if fh is not None:
        fh.close()
        api._prediction_logger._file = None


@pytest.fixture
def metrics_client(synthetic_artifact, isolated_log_path, monkeypatch):
    """Yield a TestClient backed by a synthetic model and isolated log path."""
    monkeypatch.delenv("SERVICE_GIT_COMMIT", raising=False)
    synthetic_predictor = Predictor(
        synthetic_artifact / "model.skops",
        synthetic_artifact / "model_info.json",
    )
    monkeypatch.setattr(api, "predictor", synthetic_predictor)
    with TestClient(api.app, raise_server_exceptions=False) as client:
        yield client, synthetic_predictor


def _scrape_metrics(client: TestClient) -> dict[tuple[str, frozenset[tuple[str, str]]], float]:
    response = client.get("/metrics")
    assert response.status_code == 200
    return _parse_metrics(response.text)


# ---------------------------------------------------------------------------
# 1. Exact six-metric specification & registry isolation
# ---------------------------------------------------------------------------


class TestSixMetricSpecification:
    """Assert exact names, types, labels, buckets, and absence of extra metrics."""

    def test_collector_definitions_match_exact_six_metric_spec(self):
        reg = CollectorRegistry(auto_describe=True)
        collector = APIMetrics(registry=reg)

        registered = {
            "predictions_total": collector.predictions_total,
            "prediction_risk_score": collector.prediction_risk_score,
            "prediction_probability": collector.prediction_probability,
            "model_info": collector.model_info,
            "api_requests_total": collector.api_requests_total,
            "api_request_latency_seconds": collector.api_request_latency_seconds,
        }

        for metric_name, expected_labels in EXPECTED_METRIC_LABELS.items():
            assert registered[metric_name]._labelnames == expected_labels

        assert collector.prediction_risk_score._upper_bounds == EXPECTED_RISK_SCORE_BOUNDS
        assert collector.prediction_probability._upper_bounds == EXPECTED_PROBABILITY_BOUNDS

        for forbidden in FORBIDDEN_EXTRA_METRICS:
            assert not hasattr(collector, forbidden)

        collector.set_model_info("model-a", "1", "commit-1")
        collector.record_request("POST", "/predict", 200, 0.012)
        collector.record_prediction(
            PredictionResponse(
                probability=0.72,
                risk_score=72.0,
                risk_band="High",
                model_version="2",
                disclaimer="Educational use only",
            )
        )

        text = collector.render().decode("utf-8")
        assert _app_declared_types(text) == EXPECTED_METRIC_TYPES
        families = _app_families(text)
        assert {name: fam.type for name, fam in families.items()} == EXPECTED_PARSED_FAMILY_TYPES

        assert _bucket_les(families["prediction_risk_score"]) == EXPECTED_RISK_SCORE_LE
        assert _bucket_les(families["prediction_probability"]) == EXPECTED_PROBABILITY_LE

        for forbidden in FORBIDDEN_EXTRA_METRICS:
            assert forbidden not in families
            assert forbidden not in text

    def test_metrics_endpoint_exposes_exact_six_app_metrics_labels_and_buckets(self, metrics_client):
        client, _ = metrics_client
        # Trigger at least one non-/metrics request and one successful prediction.
        assert client.get("/health").status_code == 200
        assert client.post("/predict", json={"age": 48, "sex": 1}).status_code == 200

        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert _app_declared_types(resp.text) == EXPECTED_METRIC_TYPES
        families = _app_families(resp.text)
        assert {name: fam.type for name, fam in families.items()} == EXPECTED_PARSED_FAMILY_TYPES

        for forbidden in FORBIDDEN_EXTRA_METRICS:
            assert forbidden not in families
            assert forbidden not in resp.text

        assert _bucket_les(families["prediction_risk_score"]) == EXPECTED_RISK_SCORE_LE
        assert _bucket_les(families["prediction_probability"]) == EXPECTED_PROBABILITY_LE

        # Verify exact label sets emitted on every sample of all six metrics.
        for sample in families["predictions"].samples:
            if sample.name == "predictions_total":
                assert set(sample.labels.keys()) == {"risk_band"}
                assert "model_version" not in sample.labels

        for sample in families["prediction_risk_score"].samples:
            if sample.name == "prediction_risk_score_bucket":
                assert set(sample.labels.keys()) == {"le"}
            elif sample.name in ("prediction_risk_score_count", "prediction_risk_score_sum"):
                assert sample.labels == {}

        for sample in families["prediction_probability"].samples:
            if sample.name == "prediction_probability_bucket":
                assert set(sample.labels.keys()) == {"le"}
            elif sample.name in ("prediction_probability_count", "prediction_probability_sum"):
                assert sample.labels == {}

        for sample in families["model_info"].samples:
            assert set(sample.labels.keys()) == {"model_name", "model_version", "git_commit"}

        for sample in families["api_requests"].samples:
            if sample.name == "api_requests_total":
                assert set(sample.labels.keys()) == {"endpoint", "method", "status"}

        for sample in families["api_request_latency_seconds"].samples:
            if sample.name == "api_request_latency_seconds_bucket":
                assert set(sample.labels.keys()) == {"endpoint", "le"}
                assert "method" not in sample.labels
            elif sample.name in ("api_request_latency_seconds_count", "api_request_latency_seconds_sum"):
                assert set(sample.labels.keys()) == {"endpoint"}
                assert "method" not in sample.labels

    def test_histogram_buckets_place_boundary_values_accurately(self):
        collector = APIMetrics(registry=CollectorRegistry(auto_describe=True))
        for prob, score, band in [
            (0.0, 0.0, "Low"),
            (0.72, 72.0, "High"),
            (1.0, 100.0, "High"),
        ]:
            collector.record_prediction(
                PredictionResponse(
                    probability=prob,
                    risk_score=score,
                    risk_band=band,
                    model_version="2",
                    disclaimer="Educational use only",
                )
            )

        parsed = _parse_metrics(collector.render().decode("utf-8"))
        # prediction_risk_score cumulative counts
        assert _sample_value(parsed, "prediction_risk_score_bucket", le="0.0") == 1.0
        assert _sample_value(parsed, "prediction_risk_score_bucket", le="70.0") == 1.0
        assert _sample_value(parsed, "prediction_risk_score_bucket", le="80.0") == 2.0
        assert _sample_value(parsed, "prediction_risk_score_bucket", le="100.0") == 3.0
        assert _sample_value(parsed, "prediction_risk_score_bucket", le="+Inf") == 3.0
        assert _sample_value(parsed, "prediction_risk_score_count") == 3.0
        assert _sample_value(parsed, "prediction_risk_score_sum") == pytest.approx(172.0)

        # prediction_probability cumulative counts
        assert _sample_value(parsed, "prediction_probability_bucket", le="0.0") == 1.0
        assert _sample_value(parsed, "prediction_probability_bucket", le="0.7") == 1.0
        assert _sample_value(parsed, "prediction_probability_bucket", le="0.8") == 2.0
        assert _sample_value(parsed, "prediction_probability_bucket", le="1.0") == 3.0
        assert _sample_value(parsed, "prediction_probability_bucket", le="+Inf") == 3.0
        assert _sample_value(parsed, "prediction_probability_count") == 3.0
        assert _sample_value(parsed, "prediction_probability_sum") == pytest.approx(1.72)

    def test_custom_registries_avoid_duplicate_timeseries_and_leave_global_registry_clean(self):
        for _ in range(3):
            reg = create_metrics_registry()
            collector = APIMetrics(registry=reg)
            collector.set_model_info("model-a", "1", "commit-1")
            rendered = collector.render().decode("utf-8")
            assert "model_info" in rendered
            assert "python_info" in rendered

        global_names = {family.name for family in REGISTRY.collect()}
        for metric_name in EXPECTED_METRIC_TYPES:
            assert metric_name not in global_names


# ---------------------------------------------------------------------------
# 2. Lifespan registration & model_info Gauge
# ---------------------------------------------------------------------------


class TestLifespanAndModelInfoGauge:
    """Verify repeated lifespans and model_info Gauge label handling."""

    def test_repeated_testclient_lifespans_do_not_error(self, synthetic_artifact, monkeypatch):
        monkeypatch.delenv("SERVICE_GIT_COMMIT", raising=False)
        for idx in range(4):
            monkeypatch.setenv("SERVICE_GIT_COMMIT", f"commit-{idx}")
            pred = Predictor(
                synthetic_artifact / "model.skops",
                synthetic_artifact / "model_info.json",
            )
            monkeypatch.setattr(api, "predictor", pred)
            with TestClient(api.app) as client:
                resp = client.get("/metrics")
                assert resp.status_code == 200

        parsed = _parse_metrics(resp.text)
        active_labels = _all_label_pairs(parsed, "model_info")
        assert active_labels == [
            {
                "model_name": "synthetic-cardiometabolic-model",
                "model_version": "99",
                "git_commit": "commit-3",
            }
        ]
        assert _sample_value(
            parsed,
            "model_info",
            model_name="synthetic-cardiometabolic-model",
            model_version="99",
            git_commit="commit-3",
        ) == 1.0

    def test_model_info_gauge_defaults_git_commit_to_unknown(self, metrics_client):
        client, _ = metrics_client
        parsed = _scrape_metrics(client)
        assert _sample_value(
            parsed,
            "model_info",
            model_name="synthetic-cardiometabolic-model",
            model_version="99",
            git_commit="unknown",
        ) == 1.0
        # Training commit from model_info.json must NOT be used as the git_commit label.
        assert "training-commit-should-not-be-used-in-gauge" not in client.get("/metrics").text

    def test_model_info_gauge_uses_service_git_commit_env_var(self, synthetic_artifact, monkeypatch):
        commit = "d4c3b2a1" * 5
        monkeypatch.setenv("SERVICE_GIT_COMMIT", commit)
        pred = Predictor(
            synthetic_artifact / "model.skops",
            synthetic_artifact / "model_info.json",
        )
        monkeypatch.setattr(api, "predictor", pred)

        with TestClient(api.app) as client:
            parsed = _scrape_metrics(client)
            assert _sample_value(
                parsed,
                "model_info",
                model_name="synthetic-cardiometabolic-model",
                model_version="99",
                git_commit=commit,
            ) == 1.0

    def test_failed_startup_clears_model_info_and_leaves_health_503(self, tmp_path, monkeypatch):
        broken_predictor = Predictor(tmp_path / "missing.skops", tmp_path / "missing.json")
        monkeypatch.setattr(api, "predictor", broken_predictor)

        with TestClient(api.app) as client:
            assert client.get("/health").status_code == 503
            parsed = _scrape_metrics(client)
            assert _all_label_pairs(parsed, "model_info") == []

    def test_startup_metrics_failure_does_not_break_model_loading(self, synthetic_artifact, monkeypatch):
        pred = Predictor(
            synthetic_artifact / "model.skops",
            synthetic_artifact / "model_info.json",
        )
        monkeypatch.setattr(api, "predictor", pred)
        with patch.object(api.metrics, "set_model_info", side_effect=RuntimeError("gauge failure")):
            with TestClient(api.app) as client:
                resp = client.get("/health")
                assert resp.status_code == 200
                assert resp.json()["model_loaded"] is True


# ---------------------------------------------------------------------------
# 3. /metrics endpoint, exclusion & Request-ID middleware
# ---------------------------------------------------------------------------


class TestMetricsEndpointAndExclusion:
    """Verify GET /metrics exposition, scrape exclusion, and Request-ID activity."""

    def test_metrics_endpoint_returns_prometheus_content_type(self, metrics_client):
        client, _ = metrics_client
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == CONTENT_TYPE_LATEST
        families = list(text_string_to_metric_families(response.text))
        assert len(families) > 0

    def test_get_metrics_excluded_from_request_count_and_latency(self, metrics_client):
        client, _ = metrics_client
        before = _scrape_metrics(client)
        total_req_before = _sum_samples(before, "api_requests_total")
        total_lat_before = _sum_samples(before, "api_request_latency_seconds_count")

        # Scrape /metrics multiple times including query parameters.
        for path in ("/metrics", "/metrics", "/metrics?format=prom"):
            resp = client.get(path)
            assert resp.status_code == 200

        after = _scrape_metrics(client)
        assert _sum_samples(after, "api_requests_total") == total_req_before
        assert _sum_samples(after, "api_request_latency_seconds_count") == total_lat_before
        for labels in _all_label_pairs(after, "api_requests_total"):
            assert not (labels.get("method") == "GET" and labels.get("endpoint") == "/metrics")

    def test_non_get_metrics_request_is_still_counted(self, metrics_client):
        client, _ = metrics_client
        before = _scrape_metrics(client)

        resp = client.post("/metrics")
        assert resp.status_code == 405

        after = _scrape_metrics(client)
        assert (
            _sample_value(after, "api_requests_total", endpoint="/metrics", method="POST", status="405")
            - _sample_value(before, "api_requests_total", endpoint="/metrics", method="POST", status="405")
        ) == 1.0
        assert (
            _sample_value(after, "api_request_latency_seconds_count", endpoint="/metrics")
            - _sample_value(before, "api_request_latency_seconds_count", endpoint="/metrics")
        ) == 1.0

    def test_request_id_middleware_and_logging_remain_active_for_metrics(self, metrics_client, caplog):
        client, _ = metrics_client

        # Auto-generated UUID v4 when X-Request-ID is absent.
        with caplog.at_level(logging.INFO, logger="health_api"):
            auto_resp = client.get("/metrics")
        assert auto_resp.status_code == 200
        auto_id = auto_resp.headers.get("X-Request-ID")
        assert auto_id is not None
        assert uuid.UUID(auto_id).version == 4

        metrics_logs = [
            rec.getMessage()
            for rec in caplog.records
            if rec.name == "health_api" and "request completed -" in rec.getMessage() and "path=/metrics" in rec.getMessage()
        ]
        assert any(f"request_id={auto_id}" in msg and "method=GET" in msg and "status=200" in msg for msg in metrics_logs)

        # Valid client-supplied X-Request-ID is echoed back unchanged.
        custom_id = "prom-scrape-trace-001"
        echo_resp = client.get("/metrics", headers={"X-Request-ID": custom_id})
        assert echo_resp.status_code == 200
        assert echo_resp.headers["X-Request-ID"] == custom_id

        # Invalid client-supplied X-Request-ID is replaced with UUID v4.
        bad_resp = client.get("/metrics", headers={"X-Request-ID": "bad id/with spaces"})
        assert bad_resp.status_code == 200
        assert bad_resp.headers["X-Request-ID"] != "bad id/with spaces"
        assert uuid.UUID(bad_resp.headers["X-Request-ID"]).version == 4


# ---------------------------------------------------------------------------
# 4. Bounded route-template labels & HTTP status coverage (200, 422, 500, 503)
# ---------------------------------------------------------------------------


class TestRequestMetricsAndStatusCodes:
    """Verify bounded endpoint labels and counting of 200, 405, 422, 500, and 503."""

    def test_known_endpoints_counted_with_route_templates(self, metrics_client):
        client, _ = metrics_client
        before = _scrape_metrics(client)

        health_resp = client.get("/health?probe=1")
        health_post_resp = client.post("/health")
        info_resp = client.get("/model-info")
        pred_resp = client.post("/predict", json={"age": 45, "sex": 1})

        assert health_resp.status_code == 200
        assert health_post_resp.status_code == 405
        assert info_resp.status_code == 200
        assert pred_resp.status_code == 200

        after = _scrape_metrics(client)
        assert (
            _sample_value(after, "api_requests_total", endpoint="/health", method="GET", status="200")
            - _sample_value(before, "api_requests_total", endpoint="/health", method="GET", status="200")
        ) == 1.0
        assert (
            _sample_value(after, "api_requests_total", endpoint="/health", method="POST", status="405")
            - _sample_value(before, "api_requests_total", endpoint="/health", method="POST", status="405")
        ) == 1.0
        assert (
            _sample_value(after, "api_requests_total", endpoint="/model-info", method="GET", status="200")
            - _sample_value(before, "api_requests_total", endpoint="/model-info", method="GET", status="200")
        ) == 1.0
        assert (
            _sample_value(after, "api_requests_total", endpoint="/predict", method="POST", status="200")
            - _sample_value(before, "api_requests_total", endpoint="/predict", method="POST", status="200")
        ) == 1.0

        # api_request_latency_seconds has endpoint label ONLY (combines GET + POST /health).
        assert (
            _sample_value(after, "api_request_latency_seconds_count", endpoint="/health")
            - _sample_value(before, "api_request_latency_seconds_count", endpoint="/health")
        ) == 2.0
        for endpoint in ("/model-info", "/predict"):
            assert (
                _sample_value(after, "api_request_latency_seconds_count", endpoint=endpoint)
                - _sample_value(before, "api_request_latency_seconds_count", endpoint=endpoint)
            ) == 1.0

        # Query parameters must never appear in endpoint labels.
        assert "probe=1" not in client.get("/metrics").text

    def test_unmatched_routes_use_bounded_label_never_raw_url(self, metrics_client):
        client, _ = metrics_client
        before = _scrape_metrics(client)

        raw_path = "/patients/secret-patient-id-42/record"
        resp = client.get(f"{raw_path}?bmi=33.5")
        assert resp.status_code == 404

        metrics_resp = client.get("/metrics")
        after = _parse_metrics(metrics_resp.text)
        assert (
            _sample_value(after, "api_requests_total", endpoint="unmatched", method="GET", status="404")
            - _sample_value(before, "api_requests_total", endpoint="unmatched", method="GET", status="404")
        ) == 1.0
        assert (
            _sample_value(after, "api_request_latency_seconds_count", endpoint="unmatched")
            - _sample_value(before, "api_request_latency_seconds_count", endpoint="unmatched")
        ) == 1.0
        assert "secret-patient-id-42" not in metrics_resp.text
        assert raw_path not in metrics_resp.text

    def test_422_counted_without_prediction_metrics(self, metrics_client):
        client, _ = metrics_client
        before = _scrape_metrics(client)

        resp = client.post("/predict", json={"age": 150, "sex": 1})
        assert resp.status_code == 422

        after = _scrape_metrics(client)
        assert (
            _sample_value(after, "api_requests_total", endpoint="/predict", method="POST", status="422")
            - _sample_value(before, "api_requests_total", endpoint="/predict", method="POST", status="422")
        ) == 1.0
        assert (
            _sample_value(after, "api_request_latency_seconds_count", endpoint="/predict")
            - _sample_value(before, "api_request_latency_seconds_count", endpoint="/predict")
        ) == 1.0
        assert _sum_samples(after, "predictions_total") == _sum_samples(before, "predictions_total")
        assert _sum_samples(after, "prediction_risk_score_count") == _sum_samples(before, "prediction_risk_score_count")
        assert _sum_samples(after, "prediction_probability_count") == _sum_samples(
            before, "prediction_probability_count"
        )

    def test_500_counted_without_prediction_metrics(self, metrics_client, monkeypatch):
        client, pred = metrics_client
        before = _scrape_metrics(client)

        def boom(_):
            raise RuntimeError("simulated inference crash")

        monkeypatch.setattr(pred, "predict", boom)
        resp = client.post("/predict", json={"age": 55, "sex": 1})
        assert resp.status_code == 500
        assert resp.json() == {"detail": "Prediction could not be completed"}

        after = _scrape_metrics(client)
        assert (
            _sample_value(after, "api_requests_total", endpoint="/predict", method="POST", status="500")
            - _sample_value(before, "api_requests_total", endpoint="/predict", method="POST", status="500")
        ) == 1.0
        assert (
            _sample_value(after, "api_request_latency_seconds_count", endpoint="/predict")
            - _sample_value(before, "api_request_latency_seconds_count", endpoint="/predict")
        ) == 1.0
        assert _sum_samples(after, "predictions_total") == _sum_samples(before, "predictions_total")
        assert _sum_samples(after, "prediction_risk_score_count") == _sum_samples(before, "prediction_risk_score_count")
        assert _sum_samples(after, "prediction_probability_count") == _sum_samples(
            before, "prediction_probability_count"
        )

    def test_503_counted_across_endpoints_without_prediction_metrics(self, metrics_client):
        client, pred = metrics_client
        before = _scrape_metrics(client)

        pred.is_loaded = False
        health_resp = client.get("/health")
        info_resp = client.get("/model-info")
        predict_resp = client.post("/predict", json={"age": 55, "sex": 1})

        assert health_resp.status_code == 503
        assert health_resp.json() == {
            "status": "unhealthy",
            "model_loaded": False,
            "model_version": None,
            "error": "Model is unavailable",
        }
        assert info_resp.status_code == 503
        assert predict_resp.status_code == 503

        after = _scrape_metrics(client)
        assert (
            _sample_value(after, "api_requests_total", endpoint="/health", method="GET", status="503")
            - _sample_value(before, "api_requests_total", endpoint="/health", method="GET", status="503")
        ) == 1.0
        assert (
            _sample_value(after, "api_requests_total", endpoint="/model-info", method="GET", status="503")
            - _sample_value(before, "api_requests_total", endpoint="/model-info", method="GET", status="503")
        ) == 1.0
        assert (
            _sample_value(after, "api_requests_total", endpoint="/predict", method="POST", status="503")
            - _sample_value(before, "api_requests_total", endpoint="/predict", method="POST", status="503")
        ) == 1.0
        assert _sum_samples(after, "predictions_total") == _sum_samples(before, "predictions_total")
        assert _sum_samples(after, "prediction_risk_score_count") == _sum_samples(before, "prediction_risk_score_count")
        assert _sum_samples(after, "prediction_probability_count") == _sum_samples(
            before, "prediction_probability_count"
        )


# ---------------------------------------------------------------------------
# 5. Prediction metrics, PredictionLogger preservation, and privacy
# ---------------------------------------------------------------------------


class TestPredictionMetricsLoggerAndPrivacy:
    """Verify prediction metrics, preserved JSONL logging, and zero PII/request IDs."""

    @pytest.mark.parametrize(
        "prob,expected_score,expected_band",
        [
            (0.20, 20.0, "Low"),
            (0.45, 45.0, "Moderate"),
            (0.75, 75.0, "High"),
        ],
    )
    def test_successful_prediction_records_band_score_and_probability(
        self, metrics_client, monkeypatch, prob, expected_score, expected_band
    ):
        client, pred = metrics_client
        before = _scrape_metrics(client)

        monkeypatch.setattr(
            pred.model,
            "predict_proba",
            lambda _: np.array([[1.0 - prob, prob]]),
        )
        resp = client.post("/predict", json={"age": 50, "sex": 2})
        assert resp.status_code == 200
        body = resp.json()
        assert set(body.keys()) == {"probability", "risk_score", "risk_band", "model_version", "disclaimer"}
        assert body["risk_band"] == expected_band
        assert body["risk_score"] == expected_score
        assert body["probability"] == prob

        after = _scrape_metrics(client)
        assert (
            _sample_value(after, "predictions_total", risk_band=expected_band)
            - _sample_value(before, "predictions_total", risk_band=expected_band)
        ) == 1.0
        assert (
            _sample_value(after, "prediction_risk_score_count")
            - _sample_value(before, "prediction_risk_score_count")
        ) == 1.0
        assert (
            _sample_value(after, "prediction_risk_score_sum")
            - _sample_value(before, "prediction_risk_score_sum")
        ) == pytest.approx(expected_score)
        assert (
            _sample_value(after, "prediction_probability_count")
            - _sample_value(before, "prediction_probability_count")
        ) == 1.0
        assert (
            _sample_value(after, "prediction_probability_sum")
            - _sample_value(before, "prediction_probability_sum")
        ) == pytest.approx(prob)

    def test_prediction_logger_preserved_alongside_metrics(self, metrics_client, isolated_log_path):
        client, _ = metrics_client
        custom_id = "offline-log-metrics-trace-42"
        resp = client.post(
            "/predict",
            json={"age": 55, "sex": 1, "bmi": 28.4, "smoking_status": "former"},
            headers={"X-Request-ID": custom_id},
        )
        assert resp.status_code == 200
        assert resp.headers["X-Request-ID"] == custom_id

        lines = Path(isolated_log_path).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["request_id"] == custom_id
        assert entry["inputs"]["bmi"] == 28.4
        assert entry["inputs"]["smoking_status"] == "former"
        assert len(entry["inputs"]) == 21

    def test_metrics_still_recorded_if_prediction_logger_fails(self, metrics_client):
        client, _ = metrics_client
        before = _scrape_metrics(client)

        with patch.object(api._prediction_logger, "log_prediction", side_effect=OSError("disk full")):
            resp = client.post("/predict", json={"age": 48, "sex": 1})

        assert resp.status_code == 200
        band = resp.json()["risk_band"]
        after = _scrape_metrics(client)
        assert (
            _sample_value(after, "predictions_total", risk_band=band)
            - _sample_value(before, "predictions_total", risk_band=band)
        ) == 1.0
        assert (
            _sample_value(after, "api_requests_total", endpoint="/predict", method="POST", status="200")
            - _sample_value(before, "api_requests_total", endpoint="/predict", method="POST", status="200")
        ) == 1.0

    def test_metrics_failure_never_breaks_predict_or_logging(self, metrics_client, isolated_log_path):
        client, _ = metrics_client
        with (
            patch.object(api.metrics, "record_prediction", side_effect=RuntimeError("metrics boom")),
            patch.object(api.metrics, "record_request", side_effect=RuntimeError("request metrics boom")),
        ):
            resp = client.post("/predict", json={"age": 44, "sex": 2})

        assert resp.status_code == 200
        lines = Path(isolated_log_path).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

    def test_metrics_contain_no_patient_data_or_request_ids(self, metrics_client):
        client, _ = metrics_client
        sentinel_request_id = "req-sensitive-trace-987654"
        payload = {
            "age": 67,
            "sex": 1,
            "education": 2.0,
            "income_ratio": 1.75,
            "height_cm": 173.4,
            "weight_kg": 96.2,
            "bmi": 32.0,
            "waist_cm": 109.5,
            "sbp": 152.0,
            "dbp": 94.0,
            "pulse": 84.0,
            "told_high_bp": 1.0,
            "total_chol": 248.0,
            "hdl": 38.0,
            "smoking_status": "current",
            "alcohol_drinks_week": 11.0,
            "exercise_min_week": 15.0,
            "sedentary_min_day": 540.0,
            "sleep_hours": 5.5,
            "phq9_score": 14.0,
            "phq9_category": "moderate",
        }
        resp = client.post("/predict", json=payload, headers={"X-Request-ID": sentinel_request_id})
        assert resp.status_code == 200

        metrics_text = client.get("/metrics").text
        assert sentinel_request_id not in metrics_text
        assert "request_id" not in metrics_text

        for family in text_string_to_metric_families(metrics_text):
            for sample in family.samples:
                for label_name, label_value in sample.labels.items():
                    assert label_name not in PredictionInput.model_fields
                    assert label_name != "request_id"
                    assert sentinel_request_id not in label_value
                    assert "109.5" not in label_value
                    assert "173.4" not in label_value
                    assert "current" not in label_value
