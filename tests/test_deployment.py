"""Deployment identity, validation, and secret handling without network access."""

import json
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

import pytest

from scripts import render_deploy as deploy
from scripts import smoke_test as smoke


@pytest.fixture
def deployed_service(monkeypatch):
    state = {"commit": "a" * 40, "invalid_accepted": False}
    def fetch(base_url, path, payload=None, timeout=15):
        if path == "/health":
            return json.dumps({"status": "healthy", "model_loaded": True, "model_version": "2"}).encode()
        if path == "/model-info":
            return json.dumps({"model_name": "cardiometabolic-risk-model", "model_version": "2",
                               "features": [str(i) for i in range(21)], "service_git_commit": state["commit"]}).encode()
        if path == "/docs":
            return b"<div id='swagger-ui'></div>"
        if payload.get("education") == 2.5 and not state["invalid_accepted"]:
            raise HTTPError(base_url + path, 422, "validation", {}, None)
        return json.dumps({"probability": 0.3866, "risk_score": 38.7, "risk_band": "Moderate",
                           "model_version": "2", "disclaimer": "Educational awareness only"}).encode()
    monkeypatch.setattr(smoke, "fetch", fetch)
    return state


def test_deployment_smoke_checks_identity_and_prediction(deployed_service):
    result = smoke.check_service("https://example.onrender.com", "a" * 40, "2")
    assert result["status"] == "PASS"
    assert result["service_git_commit"] == "a" * 40


def test_healthy_old_application_does_not_pass(deployed_service):
    with pytest.raises(ValueError, match="expected application commit"):
        smoke.check_service("https://example.onrender.com", "b" * 40)


def test_invalid_category_acceptance_blocks_release(deployed_service):
    deployed_service["invalid_accepted"] = True
    with pytest.raises(ValueError, match="Invalid education"):
        smoke.check_service("https://example.onrender.com")


@pytest.mark.parametrize("url", ["http://example.com", "https://user:secret@example.com",
                                "https://example.com?key=secret", "https://example.com/path"])
def test_smoke_url_rejects_credentials_and_non_origin_urls(url):
    with pytest.raises(ValueError):
        smoke.validate_base_url(url)


def test_hook_deploys_exact_sha_without_printing_secret(monkeypatch, capsys):
    captured = []
    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(deploy, "urlopen", lambda request, timeout: (captured.append(request) or Response()))
    deploy.trigger("https://api.render.com/deploy/srv-example?key=PRIVATE_HOOK", "ghcr.io/rishi799/health-wellness-api", "a" * 40)
    query = parse_qs(urlsplit(captured[0].full_url).query)
    assert query["imgURL"] == ["ghcr.io/rishi799/health-wellness-api:" + "a" * 40]
    assert "PRIVATE_HOOK" not in capsys.readouterr().out


def test_hook_failure_does_not_expose_url(monkeypatch):
    def broken(*args, **kwargs):
        raise URLError("https://api.render.com/deploy/example?key=PRIVATE_HOOK")
    monkeypatch.setattr(deploy, "urlopen", broken)
    with pytest.raises(RuntimeError) as result:
        deploy.trigger("https://api.render.com/deploy/srv-example?key=PRIVATE_HOOK", "ghcr.io/rishi799/health-wellness-api", "a" * 40)
    assert "PRIVATE_HOOK" not in str(result.value)
    assert result.value.__suppress_context__


def test_hook_without_verification_url_fails_before_remote_write(monkeypatch):
    monkeypatch.setenv("RENDER_DEPLOY_HOOK", "PRIVATE_HOOK")
    monkeypatch.delenv("RENDER_SERVICE_URL", raising=False)
    calls = []
    monkeypatch.setattr(deploy, "trigger", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="RENDER_SERVICE_URL"):
        deploy.main()
    assert not calls


def test_missing_hook_explicitly_reports_no_deployment(monkeypatch, capsys):
    monkeypatch.delenv("RENDER_DEPLOY_HOOK", raising=False)
    deploy.main()
    assert "no service was deployed" in capsys.readouterr().out
