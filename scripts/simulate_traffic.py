"""Replay a production CSV against /predict at a bounded sequential rate.

Run locally: python -m scripts.simulate_traffic --batch data/production/batches/batch_01.csv
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import ipaddress
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx
import pandas as pd

from src.api.schemas import PredictionInput

STRING_FEATURES = {"smoking_status", "phq9_category"}
INTEGER_FEATURES = {"age", "sex"}


def row_to_payload(row: pd.Series | dict) -> dict:
    """Keep serving features only and preserve invalid values for API validation."""
    payload = {}
    for field in PredictionInput.model_fields:
        value = row.get(field)
        if pd.isna(value):
            continue
        if field in STRING_FEATURES:
            payload[field] = str(value)
        elif isinstance(value, bool):
            payload[field] = value
        else:
            try:
                number = float(value)
            except (TypeError, ValueError):
                payload[field] = str(value)
                continue
            # JSON has no infinity. Preserve a readable invalid value for a 422.
            if not math.isfinite(number):
                payload[field] = str(value)
            elif field in INTEGER_FEATURES and number.is_integer():
                payload[field] = int(number)
            else:
                payload[field] = number
    return payload


def validate_api_url(value: str) -> str:
    parts = urlsplit(value)
    if (parts.scheme not in ("http", "https") or not parts.hostname or
            parts.username or parts.password or parts.query or parts.fragment or
            parts.path not in ("", "/")):
        raise ValueError("Use an HTTP(S) API origin without credentials, paths or query strings")
    return value.rstrip("/")


def is_loopback_url(value: str) -> bool:
    hostname = urlsplit(value).hostname
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def replay_batch(frame: pd.DataFrame, api_url: str, *, rate: float = 5,
                 limit: int | None = None, bad: int = 0, timeout: float = 10,
                 client=None) -> dict:
    """Send each selected row once, reporting HTTP and transport failures separately."""
    api_url = validate_api_url(api_url)
    if (not math.isfinite(rate) or rate <= 0 or not math.isfinite(timeout) or timeout <= 0 or
            (limit is not None and limit < 0) or bad < 0):
        raise ValueError("rate/timeout must be positive and finite; limit/bad must be nonnegative")
    selected = frame if limit is None else frame.head(limit)
    payloads = [row_to_payload(row) for _, row in selected.iterrows()]
    payloads.extend({"age": 55, "sex": 1, "education": 2.5} for _ in range(bad))
    if not is_loopback_url(api_url) and (len(payloads) > 20 or rate > 1):
        raise ValueError("Remote demos are limited to 20 requests at 1 request/second; use localhost for batch traffic")

    counts = {"sent": 0, "200": 0, "422": 0, "other": 0, "network_errors": 0}
    latencies = []
    next_request = time.monotonic()
    context = httpx.Client() if client is None else nullcontext(client)
    with context as session:
        for payload in payloads:
            time.sleep(max(0.0, next_request - time.monotonic()))
            started = time.monotonic()
            next_request = started + 1 / rate
            counts["sent"] += 1
            try:
                response = session.post(api_url + "/predict", json=payload, timeout=timeout,
                                        follow_redirects=False)
                category = str(response.status_code) if response.status_code in (200, 422) else "other"
                counts[category] += 1
            except httpx.RequestError:
                counts["network_errors"] += 1
            latencies.append((time.monotonic() - started) * 1000)
    return {**counts, "mean_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--bad", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args(argv)
    try:
        frame = pd.read_csv(args.batch)
        summary = replay_batch(frame, args.api_url, rate=args.rate, limit=args.limit,
                               bad=args.bad, timeout=args.timeout)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.exit(2, f"Traffic simulation failed: {type(exc).__name__}: {exc}\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 1 if summary["network_errors"] or summary["other"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
