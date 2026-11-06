"""Deploy a tested SHA image using a secret hook, then verify the public service.

Read credentials only from the environment; never print the hook or its response.
"""

import json
import os
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from scripts.smoke_test import validate_base_url, wait_for_service


def trigger(hook, image, commit):
    parts = urlsplit(hook)
    if (parts.scheme != "https" or parts.hostname != "api.render.com" or
            parts.username or parts.password or parts.fragment or not parts.path.startswith("/deploy/")):
        raise ValueError("Expected a Render HTTPS deploy hook")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Release commit must be a full lowercase Git SHA")
    if image != "ghcr.io/rishi799/health-wellness-api":
        raise ValueError("Unexpected image repository")
    query = dict(parse_qsl(parts.query))
    if not query.get("key"):
        raise ValueError("Render hook has no key")
    query["imgURL"] = f"{image}:{commit}"
    url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))
    try:
        with urlopen(Request(url, data=b"", method="POST"), timeout=30) as response:
            if response.status not in (200, 202):
                raise RuntimeError("Render did not accept the deploy request")
    except Exception:
        # urllib exceptions can contain the full secret URL. Drop that context.
        raise RuntimeError("Render deploy request failed; inspect Render privately") from None


def main():
    hook = os.environ.get("RENDER_DEPLOY_HOOK", "")
    base_url = os.environ.get("RENDER_SERVICE_URL", "")
    if not hook:
        print("Render deployment not configured: image publication only; no service was deployed.")
        return
    if not base_url:
        raise ValueError("Set repository variable RENDER_SERVICE_URL before enabling the deploy hook")
    validate_base_url(base_url)
    if urlsplit(base_url).scheme != "https":
        raise ValueError("RENDER_SERVICE_URL must be the public HTTPS service origin")
    commit = os.environ["RELEASE_SHA"]
    trigger(hook, os.environ.get("IMAGE", "ghcr.io/rishi799/health-wellness-api"), commit)
    print("Render accepted the SHA-pinned image. Waiting for the new service identity.")
    print(json.dumps(wait_for_service(base_url, expected_commit=commit, attempts=60, delay=10, timeout=20)))


if __name__ == "__main__":
    main()
