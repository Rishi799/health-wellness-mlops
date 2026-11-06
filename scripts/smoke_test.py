"""Check a local container or public deployment using synthetic inputs only.

Usage: python -m scripts.smoke_test --base-url https://SERVICE.onrender.com
"""

import argparse
import json
import math
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def validate_base_url(value):
    parts = urlsplit(value)
    if (not parts.hostname or parts.username or parts.password or parts.query or
            parts.fragment or parts.path not in ("", "/") or
            not (parts.scheme == "https" or
                 (parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1", "::1")))):
        raise ValueError("Use an HTTPS service origin, or an HTTP localhost origin")
    return value.rstrip("/")


def fetch(base_url, path, payload=None, timeout=15):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(base_url + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def check_service(base_url, expected_commit=None, expected_model_version=None, timeout=15):
    """Verify identity and response content, rather than merely HTTP status 200."""
    base_url = validate_base_url(base_url)
    health = json.loads(fetch(base_url, "/health", timeout=timeout))
    if health.get("status") != "healthy" or health.get("model_loaded") is not True:
        raise ValueError("Model is not healthy")
    info = json.loads(fetch(base_url, "/model-info", timeout=timeout))
    features = info.get("features", [])
    if (info.get("model_name") != "cardiometabolic-risk-model" or
            len(features) != 21 or len(set(features)) != 21):
        raise ValueError("Unexpected model or feature contract")
    if expected_commit and info.get("service_git_commit") != expected_commit:
        raise ValueError("Service has not switched to the expected application commit")
    if expected_model_version and info.get("model_version") != expected_model_version:
        raise ValueError("Unexpected model version")
    result = json.loads(fetch(base_url, "/predict", {"age": 55, "sex": 1, "bmi": 30.0,
                                                    "sbp": 140.0, "told_high_bp": 1.0}, timeout))
    probability, score = result["probability"], result["risk_score"]
    if not (math.isfinite(probability) and 0 <= probability <= 1 and math.isfinite(score) and 0 <= score <= 100):
        raise ValueError("Invalid prediction values")
    if abs(score - probability * 100) > 0.056:
        raise ValueError("Probability and score disagree beyond documented rounding")
    band = "Low" if score < 30 else "Moderate" if score < 60 else "High"
    if result.get("risk_band") != band or not result.get("disclaimer"):
        raise ValueError("Missing disclaimer or incorrect display band")
    if health.get("model_version") != info.get("model_version") or result.get("model_version") != info.get("model_version"):
        raise ValueError("Endpoint model versions disagree")
    if b"swagger-ui" not in fetch(base_url, "/docs", timeout=timeout).lower():
        raise ValueError("Interactive API documentation is unavailable")
    try:
        fetch(base_url, "/predict", {"age": 55, "sex": 1, "education": 2.5}, timeout)
    except HTTPError as exc:
        if exc.code != 422:
            raise ValueError("Invalid category did not return 422") from None
    else:
        raise ValueError("Invalid education category was accepted")
    return {"status": "PASS", "model_name": info["model_name"], "model_version": info["model_version"],
            "service_git_commit": info.get("service_git_commit", "unknown")}


def wait_for_service(base_url, expected_commit=None, expected_model_version=None,
                     attempts=30, delay=2, timeout=15):
    error = None
    for attempt in range(attempts):
        try:
            return check_service(base_url, expected_commit, expected_model_version, timeout)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(delay)
    # Do not emit response bodies or URLs from transport exceptions.
    raise RuntimeError(f"Service checks failed after {attempts} attempts ({type(error).__name__})") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-model-version")
    parser.add_argument("--attempts", type=int, default=30)
    parser.add_argument("--delay", type=float, default=2)
    parser.add_argument("--timeout", type=float, default=15)
    args = parser.parse_args(argv)
    if args.attempts < 1 or args.delay < 0 or args.timeout <= 0:
        parser.error("attempts must be positive, delay nonnegative, and timeout positive")
    print(json.dumps(wait_for_service(args.base_url, args.expected_commit, args.expected_model_version,
                                      args.attempts, args.delay, args.timeout), indent=2))


if __name__ == "__main__":
    main()
