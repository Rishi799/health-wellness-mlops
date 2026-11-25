"""Offline tests for the Task 3 monitoring stack configuration.

Validates:
1. docker-compose.yml services, image pins, volumes, env vars.
2. prometheus.yml scrape target and interval.
3. Grafana datasource provisioning connects to http://prometheus:9090.
4. Grafana dashboard provisioning loads from /var/lib/grafana/dashboards.
5. Dashboard JSON has seven panels with correct titles, datasource variable,
   and valid PromQL expressions matching the required specification.
6. No secrets are committed (GRAFANA_ADMIN_PASSWORD uses env var without fallback).
"""

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. docker-compose.yml
# ---------------------------------------------------------------------------


class TestDockerCompose:
    """Validate the Docker Compose monitoring stack definition."""

    @pytest.fixture(scope="class")
    def compose(self):
        with open(ROOT / "docker-compose.yml") as f:
            return yaml.safe_load(f)

    @pytest.fixture(scope="class")
    def compose_text(self):
        return (ROOT / "docker-compose.yml").read_text()

    def test_has_four_services(self, compose):
        expected = {"init", "api", "prometheus", "grafana"}
        assert set(compose["services"].keys()) == expected

    def test_api_preserves_init_dependency(self, compose):
        api = compose["services"]["api"]
        assert api["depends_on"]["init"]["condition"] == "service_completed_successfully"

    def test_api_preserves_logs_volume(self, compose):
        api = compose["services"]["api"]
        assert "./logs:/app/logs" in api["volumes"]

    def test_api_preserves_non_root_healthcheck(self, compose):
        api = compose["services"]["api"]
        assert "healthcheck" in api

    def test_prometheus_image_pinned(self, compose):
        image = compose["services"]["prometheus"]["image"]
        assert image.startswith("prom/prometheus:v")
        assert "latest" not in image

    def test_grafana_image_pinned(self, compose):
        image = compose["services"]["grafana"]["image"]
        assert image.startswith("grafana/grafana:")
        assert "latest" not in image

    def test_prometheus_mounts_config_readonly(self, compose):
        volumes = compose["services"]["prometheus"]["volumes"]
        assert any("prometheus.yml:/etc/prometheus/prometheus.yml:ro" in v for v in volumes)

    def test_grafana_has_named_volume(self, compose):
        volumes = compose["services"]["grafana"]["volumes"]
        assert any("grafana-data:/var/lib/grafana" in v for v in volumes)
        assert "grafana-data" in compose.get("volumes", {})

    def test_grafana_admin_password_uses_env_var_without_fallback(self, compose_text):
        """The Grafana admin password must reference $GRAFANA_ADMIN_PASSWORD
        with no :- default fallback."""
        assert "${GRAFANA_ADMIN_PASSWORD}" in compose_text
        assert ":-" not in compose_text.split("GRAFANA_ADMIN_PASSWORD")[1].split("\n")[0]

    def test_grafana_mounts_provisioning_readonly(self, compose):
        volumes = compose["services"]["grafana"]["volumes"]
        assert any("provisioning:/etc/grafana/provisioning:ro" in v for v in volumes)

    def test_grafana_mounts_dashboards_readonly(self, compose):
        volumes = compose["services"]["grafana"]["volumes"]
        assert any("dashboards:/var/lib/grafana/dashboards:ro" in v for v in volumes)

    def test_prometheus_depends_on_api_healthy(self, compose):
        deps = compose["services"]["prometheus"]["depends_on"]
        assert deps["api"]["condition"] == "service_healthy"

    def test_grafana_depends_on_prometheus(self, compose):
        deps = compose["services"]["grafana"]["depends_on"]
        assert "prometheus" in deps


# ---------------------------------------------------------------------------
# 2. prometheus.yml
# ---------------------------------------------------------------------------


class TestPrometheusConfig:
    """Validate the Prometheus scrape configuration."""

    @pytest.fixture(scope="class")
    def prom_config(self):
        with open(ROOT / "monitoring" / "prometheus" / "prometheus.yml") as f:
            return yaml.safe_load(f)

    def test_scrape_interval_is_5s(self, prom_config):
        assert prom_config["global"]["scrape_interval"] == "5s"

    def test_scrapes_api_on_port_8000(self, prom_config):
        jobs = prom_config["scrape_configs"]
        api_job = next(j for j in jobs if j["job_name"] == "health-wellness-api")
        targets = api_job["static_configs"][0]["targets"]
        assert "api:8000" in targets

    def test_scrapes_metrics_path(self, prom_config):
        jobs = prom_config["scrape_configs"]
        api_job = next(j for j in jobs if j["job_name"] == "health-wellness-api")
        assert api_job.get("metrics_path", "/metrics") == "/metrics"


# ---------------------------------------------------------------------------
# 3. Grafana datasource provisioning
# ---------------------------------------------------------------------------


class TestGrafanaDatasource:
    """Validate the Grafana Prometheus datasource configuration."""

    @pytest.fixture(scope="class")
    def datasource(self):
        path = ROOT / "monitoring" / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
        with open(path) as f:
            return yaml.safe_load(f)

    def test_prometheus_url(self, datasource):
        ds = datasource["datasources"][0]
        assert ds["url"] == "http://prometheus:9090"

    def test_is_default_datasource(self, datasource):
        ds = datasource["datasources"][0]
        assert ds["isDefault"] is True

    def test_datasource_type_is_prometheus(self, datasource):
        ds = datasource["datasources"][0]
        assert ds["type"] == "prometheus"


# ---------------------------------------------------------------------------
# 4. Grafana dashboard provisioning
# ---------------------------------------------------------------------------


class TestGrafanaDashboardProvisioning:
    """Validate the dashboard provider configuration."""

    @pytest.fixture(scope="class")
    def provider(self):
        path = ROOT / "monitoring" / "grafana" / "provisioning" / "dashboards" / "dashboards.yml"
        with open(path) as f:
            return yaml.safe_load(f)

    def test_provider_loads_from_dashboards_dir(self, provider):
        p = provider["providers"][0]
        assert p["options"]["path"] == "/var/lib/grafana/dashboards"

    def test_provider_type_is_file(self, provider):
        p = provider["providers"][0]
        assert p["type"] == "file"


# ---------------------------------------------------------------------------
# 5. Dashboard JSON — seven panels matching required specification
# ---------------------------------------------------------------------------


EXPECTED_PANEL_TITLES = [
    "Requests per minute",
    "4xx/5xx error rate",
    "Request latency p50 and p95",
    "Predictions by risk band",
    "Average risk score over time",
    "Risk score histogram",
    "Serving model version",
]


class TestDashboardJSON:
    """Validate the Grafana dashboard JSON definition."""

    @pytest.fixture(scope="class")
    def dashboard(self):
        path = ROOT / "monitoring" / "grafana" / "dashboards" / "health-wellness-api.json"
        with open(path) as f:
            return json.load(f)

    def test_valid_json(self):
        """Dashboard JSON must be parseable without errors."""
        path = ROOT / "monitoring" / "grafana" / "dashboards" / "health-wellness-api.json"
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "panels" in data

    def test_has_seven_panels(self, dashboard):
        assert len(dashboard["panels"]) == 7

    def test_panel_titles_match(self, dashboard):
        titles = [p["title"] for p in dashboard["panels"]]
        assert titles == EXPECTED_PANEL_TITLES

    def test_all_panels_reference_prometheus_datasource_variable(self, dashboard):
        for panel in dashboard["panels"]:
            ds = panel["datasource"]
            assert ds["uid"] == "${DS_PROMETHEUS}", f"Panel '{panel['title']}' missing datasource variable"

    def test_all_panels_have_valid_promql_targets(self, dashboard):
        for panel in dashboard["panels"]:
            targets = panel.get("targets", [])
            assert len(targets) >= 1, f"Panel '{panel['title']}' has no targets"
            for target in targets:
                expr = target.get("expr", "")
                assert len(expr) > 0, f"Panel '{panel['title']}' has empty PromQL expression"

    # --- Panel 1: Requests per minute ---

    def test_requests_per_minute_uses_api_requests_total(self, dashboard):
        panel = dashboard["panels"][0]
        expr = panel["targets"][0]["expr"]
        assert "api_requests_total" in expr

    def test_requests_per_minute_multiplies_by_60(self, dashboard):
        panel = dashboard["panels"][0]
        expr = panel["targets"][0]["expr"]
        assert "* 60" in expr

    def test_requests_per_minute_groups_by_endpoint(self, dashboard):
        panel = dashboard["panels"][0]
        expr = panel["targets"][0]["expr"]
        assert "by (endpoint)" in expr

    # --- Panel 2: 4xx/5xx error rate ---

    def test_error_rate_uses_4xx_5xx_status_regex(self, dashboard):
        panel = dashboard["panels"][1]
        expr = panel["targets"][0]["expr"]
        assert 'status=~"4..|5.."' in expr

    def test_error_rate_uses_5m_window(self, dashboard):
        panel = dashboard["panels"][1]
        expr = panel["targets"][0]["expr"]
        assert "[5m]" in expr

    def test_error_rate_computes_percentage(self, dashboard):
        panel = dashboard["panels"][1]
        expr = panel["targets"][0]["expr"]
        assert "100 *" in expr

    # --- Panel 3: Request latency p50 and p95 ---

    def test_latency_has_exactly_two_targets(self, dashboard):
        panel = dashboard["panels"][2]
        assert len(panel["targets"]) == 2

    def test_latency_panel_has_p50(self, dashboard):
        panel = dashboard["panels"][2]
        exprs = [t["expr"] for t in panel["targets"]]
        assert any("histogram_quantile(0.50" in e for e in exprs)

    def test_latency_panel_has_p95(self, dashboard):
        panel = dashboard["panels"][2]
        exprs = [t["expr"] for t in panel["targets"]]
        assert any("histogram_quantile(0.95" in e for e in exprs)

    def test_latency_panel_has_no_p99(self, dashboard):
        panel = dashboard["panels"][2]
        exprs = [t["expr"] for t in panel["targets"]]
        assert not any("0.99" in e for e in exprs)

    def test_latency_uses_5m_window(self, dashboard):
        panel = dashboard["panels"][2]
        for target in panel["targets"]:
            assert "[5m]" in target["expr"]

    def test_latency_groups_by_le_and_endpoint(self, dashboard):
        panel = dashboard["panels"][2]
        for target in panel["targets"]:
            assert "by (le, endpoint)" in target["expr"]

    # --- Panel 4: Predictions by risk band ---

    def test_predictions_uses_predictions_total(self, dashboard):
        panel = dashboard["panels"][3]
        assert "predictions_total" in panel["targets"][0]["expr"]

    def test_predictions_groups_by_risk_band(self, dashboard):
        panel = dashboard["panels"][3]
        assert "by (risk_band)" in panel["targets"][0]["expr"]

    def test_predictions_uses_5m_window(self, dashboard):
        panel = dashboard["panels"][3]
        assert "[5m]" in panel["targets"][0]["expr"]

    # --- Panel 5: Average risk score over time ---

    def test_avg_risk_score_uses_sum_divided_by_count(self, dashboard):
        panel = dashboard["panels"][4]
        expr = panel["targets"][0]["expr"]
        assert "prediction_risk_score_sum" in expr
        assert "prediction_risk_score_count" in expr
        assert "/" in expr

    def test_avg_risk_score_uses_5m_window(self, dashboard):
        panel = dashboard["panels"][4]
        assert "[5m]" in panel["targets"][0]["expr"]

    # --- Panel 6: Risk score histogram ---

    def test_risk_score_histogram_uses_bucket_metric(self, dashboard):
        panel = dashboard["panels"][5]
        assert "prediction_risk_score_bucket" in panel["targets"][0]["expr"]

    def test_risk_score_histogram_groups_by_le(self, dashboard):
        panel = dashboard["panels"][5]
        assert "by (le)" in panel["targets"][0]["expr"]

    # --- Panel 7: Serving model version ---

    def test_model_version_uses_model_info_metric(self, dashboard):
        panel = dashboard["panels"][6]
        assert "model_info" in panel["targets"][0]["expr"]

    def test_model_version_is_table_type(self, dashboard):
        panel = dashboard["panels"][6]
        assert panel["type"] == "table"

    def test_model_version_shows_expected_labels(self, dashboard):
        """The model_info table must expose model_name, model_version, and git_commit."""
        panel = dashboard["panels"][6]
        transforms = panel.get("transformations", [])
        organize = next((t for t in transforms if t["id"] == "organize"), None)
        assert organize is not None
        renamed = organize["options"]["renameByName"]
        assert "model_name" in renamed
        assert "model_version" in renamed
        assert "git_commit" in renamed

    def test_model_version_uses_instant_query(self, dashboard):
        panel = dashboard["panels"][6]
        assert panel["targets"][0].get("instant") is True

    # --- Dashboard-level checks ---

    def test_dashboard_has_uid(self, dashboard):
        assert dashboard["uid"] == "health-wellness-api"

    def test_dashboard_has_templating_variable_for_datasource(self, dashboard):
        variables = dashboard["templating"]["list"]
        ds_vars = [v for v in variables if v["name"] == "DS_PROMETHEUS"]
        assert len(ds_vars) == 1
        assert ds_vars[0]["type"] == "datasource"


# ---------------------------------------------------------------------------
# 6. No secrets committed
# ---------------------------------------------------------------------------


class TestNoSecretsCommitted:
    """Ensure secrets are not hardcoded in committed files."""

    def test_docker_compose_has_no_password_fallback(self):
        """The compose file must use ${GRAFANA_ADMIN_PASSWORD} with no :- default."""
        text = (ROOT / "docker-compose.yml").read_text()
        assert "${GRAFANA_ADMIN_PASSWORD}" in text
        # Ensure no fallback default
        for line in text.splitlines():
            if "GRAFANA_ADMIN_PASSWORD" in line:
                assert ":-" not in line, f"Found fallback default in: {line}"

    def test_env_example_has_placeholder_not_real_password(self):
        text = (ROOT / ".env.example").read_text()
        assert "GRAFANA_ADMIN_PASSWORD" in text
        # The placeholder value must not be a usable real password
        for line in text.splitlines():
            if line.startswith("GRAFANA_ADMIN_PASSWORD="):
                value = line.split("=", 1)[1]
                assert value.upper() == value or "CHANGE" in value.upper(), (
                    f"Placeholder looks like a real password: {value!r}"
                )

    def test_env_file_gitignored(self):
        text = (ROOT / ".gitignore").read_text()
        assert ".env" in text
