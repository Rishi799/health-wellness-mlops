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


# ---------------------------------------------------------------------------
# Dockerfile CMD — must honour Render PORT env var (Phase 2 requirement)
# ---------------------------------------------------------------------------


def _read_dockerfile_cmd() -> str:
    """Return the last CMD line from the project Dockerfile as a raw string."""
    import pathlib
    dockerfile = pathlib.Path(__file__).parent.parent / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")
    for line in reversed(text.splitlines()):
        if line.strip().startswith("CMD"):
            return line
    raise AssertionError("No CMD instruction found in Dockerfile")


def test_dockerfile_cmd_honours_port_env_var():
    """CMD must use ${PORT:-8000} so Render's PORT variable is respected."""
    cmd_line = _read_dockerfile_cmd()
    assert "${PORT:-8000}" in cmd_line, (
        f"CMD does not contain '${'{PORT:-8000}'}': {cmd_line!r}"
    )


def test_dockerfile_cmd_does_not_hardcode_port_8000():
    """CMD must not hard-code '--port', '8000' as separate JSON array elements."""
    cmd_line = _read_dockerfile_cmd()
    # The old broken form used a JSON array with literal "--port" and "8000" items.
    assert '"--port", "8000"' not in cmd_line, (
        f"CMD hard-codes --port 8000 as separate array elements: {cmd_line!r}"
    )
    assert '"--port","8000"' not in cmd_line, (
        f"CMD hard-codes --port 8000 as separate array elements: {cmd_line!r}"
    )


def test_dockerfile_cmd_exact_syntax():
    """CMD must match the exact dynamic port instruction requested by Rishi."""
    cmd_line = _read_dockerfile_cmd().strip()
    expected = 'CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port \\"${PORT:-8000}\\""]'
    assert cmd_line == expected, f"CMD instruction mismatch.\nExpected: {expected}\nActual:   {cmd_line}"


def test_port_expansion_default_and_custom():
    """Shell expansion of ${PORT:-8000} defaults to 8000 and respects custom PORT."""
    import subprocess
    # Default (no PORT set)
    res_default = subprocess.run(
        ["sh", "-c", 'echo "${PORT:-8000}"'],
        capture_output=True,
        text=True,
        env={},
    )
    assert res_default.stdout.strip() == "8000"

    # Custom PORT (e.g. Render dynamic port)
    res_custom = subprocess.run(
        ["sh", "-c", 'echo "${PORT:-8000}"'],
        capture_output=True,
        text=True,
        env={"PORT": "10000"},
    )
    assert res_custom.stdout.strip() == "10000"


def test_entrypoint_preserves_exec_passthrough():
    """entrypoint.sh must end with exec \"$@\" to pass through CMD arguments."""
    import pathlib
    entrypoint = pathlib.Path(__file__).parent.parent / "entrypoint.sh"
    content = entrypoint.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]
    assert lines[-1] == 'exec "$@"', f"entrypoint.sh does not end with exec \"$@\": {lines[-1]!r}"
