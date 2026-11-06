# Docker Guide

## Prerequisites

- Docker Desktop installed and running
- DVC data pulled (`dvc pull`) so `models/` has the exported model

## Quick Start

```bash
# Build and run with docker compose
docker compose up --build

# API docs at:
# http://localhost:8000/docs
```

## Manual Build and Run

```bash
# Build the image
docker build -t health-api .

# Run the container
docker run --rm -p 8000:8000 health-api

# Test the healthcheck
curl http://localhost:8000/health
```

## Image Details

| Property | Value |
|----------|-------|
| Base image | `python:3.11-slim` |
| Target size | < 600 MB |
| Target cold start | < 10 seconds |
| Exposed port | 8000 |
| Run user | `appuser` (non-root) |
| Requirements | `requirements-api.txt` (7 packages) |

## How the Model Gets Into the Image

1. `dvc pull models/model.skops.dvc` downloads the binary; `models/model_info.json` comes from Git
2. `docker build` copies exactly those two files into the image and removes model-directory write permissions
3. The running container loads the model at startup — no network or credentials needed at runtime

## Why `requirements-api.txt` Instead of `requirements.txt`

The full `requirements.txt` has 254 packages (Jupyter, profiling, DVC, etc.) and would make the image over 2 GB. The API only needs 7 packages to serve predictions.

## Dockerfile Layer Caching

Requirements are copied and installed **before** the source code. This means that rebuilding after a code change only re-runs the `COPY src/` layer, not the slow `pip install` layer.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `models/model.skops` not found | Run `dvc pull` before `docker build` |
| Port 8000 already in use | Stop the other service or change the port: `-p 9000:8000` |
| Image too large | Check that `.dockerignore` excludes `data/`, `.venv`, `notebooks/` |
| Container exits immediately | Check logs: `docker logs <container_id>` |
| `/health` returns 503 | Model failed to load — check model file exists in `models/` |

## Security

- Container runs as non-root user `appuser`
- No `.env`, tokens, or credentials are copied into the image
- Model files are read-only inside the container

## Render and release verification

The container honors `PORT` (default 8000). Build with `--build-arg BUILD_SHA=<full-commit>` to expose the application identity as `service_git_commit` from `/model-info`. CI/CD sets this automatically. It is separate from the model's training commit.

Follow [the Render runbook](render-deployment.md) for public hosting, a synthetic deployment check, private deploy-hook configuration, and image rollback. Image size and startup-time values above are targets until measured for the selected release.
