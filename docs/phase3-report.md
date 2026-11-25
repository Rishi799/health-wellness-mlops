# Phase 3 implementation report

## Rhea (P3) - model review, governance and fairness

The existing retraining and champion-comparison implementation was merged in
[PR #37](https://github.com/Rishi799/health-wellness-mlops/pull/37). This review
builds on that work rather than refitting or registering a duplicate candidate.
The [training record](../reports/retraining/training.json) uses 4,428 original
and 6,226 later-cycle fitting rows; the 1,108 original and 1,557 later holdout
rows are excluded. Imported preprocessing/Logistic Regression settings, split
seed 42, participant-separation checks and input/artifact fingerprints connect
the candidate to its evaluation evidence.

The [comparison](../reports/retraining/comparison.json) gives later-holdout
ROC-AUC 0.8457 for the challenger versus 0.8401 for the champion, with equal
recall 0.8191. Challenger original-holdout ROC-AUC is 0.8495 and recall 0.8062,
compared with champion recall 0.8270. The [decision](../reports/retraining/promotion_decision.json)
passes the fixed numerical eligibility rule, registers version 3 as
`challenger`, requires review and records no champion-alias change. The saved
[remote audit](../reports/phase3/remote_verification.json) records version 2 as
champion; this is dated evidence, not a fresh remote check.

Rhea's recommendation is to retain the exported champion while the team reviews
the original-population recall loss and unresolved subgroup risks. Young-adult
detection on the later holdout improves from 1/14 to 3/14; it is still poor and
uncertain. The [model card](model_card.md) links the underlying Fairlearn and
comparison evidence and the open investigation in
[issue #25](https://github.com/Rishi799/health-wellness-mlops/issues/25).

The new [governance document](governance.md) assigns owners and separates
automated eligibility from team release approval. It covers Git/DVC/MLflow/GHCR
versioning, audit evidence, data privacy, pending retention/access decisions,
threshold change control and the existing verified rollback procedure. Live
prediction logging, metrics, dashboards and scheduled alerting must not be
claimed from offline reports; their integration remains with the named owners.
Teammate review of the card/policy and a team decision on any model replacement
remain required.

Rhea's [fairness follow-up](../reports/retraining/fairness.json) consolidates all
24 model/holdout/sex-or-age entries with n, positives, detected, missed, recall
and descriptive Wilson intervals. The aggregator rejects mixed source runs,
holdout fingerprints, model fingerprints, thresholds, counts or intervals;
14 offline regression tests cover those paths. It validates saved aggregate
evidence without claiming to refit or freshly re-evaluate the candidate. The
[model card](model_card.md#fairness-follow-up-for-issue-25---22-november-2026)
includes both subgroup tables and interpretation. The later candidate still
misses 11 of 14 young-adult positives, so the follow-up does not resolve issue
#25 or authorize changing the model/threshold. The full suite passes 270 local
tests after this addition; teammate review and CI remain required.

## Rishabh — production data, monitoring, trigger and challenger handoff

The later NHANES cycle produces 7,783 validated adult records with the original 31-column contract. Seed-42 partitioning gives six disjoint simulated production batches; a separate, explicitly synthetic older-adult batch adds clipped BMI/SBP offsets. A full DVC rebuild preserved all seven original-cycle CSV fingerprints. Data, splits and the candidate artifact are uploaded through DVC.

The traffic simulator sends serving features only, respects a configurable rate and reports HTTP/transport outcomes. The 21 November audit validated all 9,081 batch/stress payloads and replayed all 1,298 rows of the first batch locally: 1,298 HTTP 200 responses, followed by five expected 422s for a separate deliberately invalid run. The API currently lacks the prediction JSONL writer, so the strict traffic → log → live drift acceptance check remains pending Niranjan's integration and a subsequent Rishabh replay.

Evidently found input drift in 8/21 features across the full later cycle. Its ROC-AUC was 0.8248 and recall 0.7846. The three-rule trigger therefore requested retraining for data drift. The synthetic stress batch triggered drift and the ROC-AUC floor; an identical-data negative control requested no retraining. Reports preserve ordinary-batch drift rather than changing the rule to force an expected demonstration.

The challenger fits 10,654 rows, excluding both holdouts. On the later holdout it achieves ROC-AUC 0.8457 versus champion 0.8401, with equal recall 0.8191. On the original holdout it achieves ROC-AUC 0.8495 and recall 0.8062. All prescribed numerical promotion checks pass. The candidate is registered as version 3 with alias `challenger`; the `champion` alias remains version 2. Updating the serving model requires the team's reviewed release process.

Fairlearn shows that retraining does not resolve young-adult performance: the challenger detects 3/14 positive cases aged 20–39 on the later holdout, compared with 1/14 for the champion. Small positive counts make the estimate uncertain. Original-population recall also decreases. These limitations remain explicit release-review items. SHAP explains the fitted Logistic Regression in log-odds, with contributions aggregated to the 21 inputs and reconstruction error around `1e-15`; the attributions do not establish causal health effects.

The [successful DagsHub run](https://dagshub.com/Rishi799/health-wellness-mlops.mlflow/#/experiments/2/runs/893b1b05690d4870ba99b0628355395f) contains candidate parameters, both holdout metrics, model artifact and aggregate review evidence. Git tracks code, tests, DVC pointers and summary reports. MLflow tracks the run/registry; DVC tracks data/model bytes; GHCR and Render retain their Phase 2 release roles. The [runbook](phase3-p2.md) provides commands, exact results and source references.

Delivery is stacked across PRs [28](https://github.com/Rishi799/health-wellness-mlops/pull/28), [29](https://github.com/Rishi799/health-wellness-mlops/pull/29), [30](https://github.com/Rishi799/health-wellness-mlops/pull/30), [31](https://github.com/Rishi799/health-wellness-mlops/pull/31), and [32](https://github.com/Rishi799/health-wellness-mlops/pull/32). Team review/merges, Niranjan's live JSONL and dashboard integration, Rishi's monitoring CI step and Rhea's model-release review remain separate integration steps. This report does not claim the whole team's Phase 3 is complete.

The [completion audit](phase3-verification.md) maps every handbook requirement to evidence and pending owners. Independent recalculation reproduced the reported metrics exactly and verified data/model hashes, split reproduction, participant separation and Fairlearn/SHAP provenance. The audit also fixed stale report outputs after failed monitoring reruns. All **256 local tests** pass; DVC local/remote status and dependency consistency checks pass. Both trigger outcomes and the Evidently report were captured in [screenshots](../reports/phase3/screenshots/), and the full champion degradation table includes precision, Brier and subgroup recall. The ordinary-batch drift differs from the handbook's initial hoped-for story; it is retained as an observed finding for review.

## Niranjan (P4) — prediction logging, Prometheus metrics, Grafana observability, and verification

Following the challenger and governance foundations established by Rhea and Rishabh, Niranjan implemented the API observability and local monitoring stack. This links live FastAPI inference to request-ID tracing, structured JSONL prediction logging, Prometheus metrics exposition, Grafana visualization, and a four-service Docker Compose setup.

### Implementation breakdown and PR milestones

The implementation was delivered across three pull requests:

1. **[PR #33](https://github.com/Rishi799/health-wellness-mlops/pull/33) — Prediction logging and request-ID middleware (`p4-prediction-logging`)**:
   - Implemented `request_id_and_latency_middleware` in `src/api/main.py`: validates incoming `X-Request-ID` headers against `^[A-Za-z0-9\-_]{1,128}$` or generates a UUID v4, sets `X-Request-ID` on every response, and logs `method`, `path`, `status`, `latency_ms`, and `request_id` via `health_api` (`StreamHandler` to stdout/stderr) without logging patient input data.
   - Built `PredictionLogger` in `src/api/monitoring.py`: a singleton-style logger with lazy directory/file creation (`_ensure_file()` on first write) and an in-process `threading.Lock` (`self.file_lock`) that synchronously appends and flushes one compact JSON line per successful `POST /predict` (`HTTP 200`) to `logs/predictions.jsonl` (configurable via `PREDICTION_LOG_PATH`). Validation errors (`422`) and failures (`500`/`503`) do not invoke `PredictionLogger`.
   - Conforms to the Phase 3 JSONL contract: records `timestamp` (UTC ISO-8601 `+00:00`), `request_id`, `model_version`, `latency_ms` (inference duration in ms), `inputs` (all 21 `PredictionInput` features, with omitted optional fields serialized as `null`), `probability`, `risk_score`, and `risk_band`.
   - Error isolation and container permissions: `PredictionLogger.log_prediction()` and its call site in `src/api/main.py` wrap file I/O in `try ... except Exception` blocks so write failures log a warning without failing startup or altering responses. Added `entrypoint.sh` to check write access to `PREDICTION_LOG_DIR` (default `/app/logs`) before starting `uvicorn` as non-root `appuser` (UID 1000).

2. **[PR #41](https://github.com/Rishi799/health-wellness-mlops/pull/41) — Prometheus API metrics exposition (`p4-metrics-dashboard`)**:
   - Implemented `APIMetrics` and `create_metrics_registry()` in `src/api/monitoring.py` using an isolated `CollectorRegistry` (plus standard `ProcessCollector`, `PlatformCollector`, and `GCCollector` runtime collectors) and exposed `GET /metrics` in Prometheus text exposition format.
   - Implemented all six application metrics:
     1. `api_requests_total` (`Counter`, labels: `endpoint`, `method`, `status`): counts HTTP requests using bounded FastAPI route templates (or `"unmatched"`), excluding `GET /metrics` scrapes.
     2. `api_request_latency_seconds` (`Histogram`, labels: `endpoint`, 11 buckets from `0.005`s to `10.0`s): tracks HTTP request latency, also excluding `GET /metrics`.
     3. `predictions_total` (`Counter`, labels: `risk_band`): counts successful predictions by `Low`, `Moderate`, or `High`.
     4. `prediction_risk_score` (`Histogram`, buckets `0` to `100` in steps of `10`): records predicted risk score distributions.
     5. `prediction_probability` (`Histogram`, buckets `0.0` to `1.0` in steps of `0.1`): records predicted probability distributions.
     6. `model_info` (`Gauge`, labels: `model_name`, `model_version`, `git_commit`): set to `1` during lifespan model loading (`git_commit` from `SERVICE_GIT_COMMIT`, defaulting to `"unknown"`) and cleared on unload or reload.

3. **[PR #42](https://github.com/Rishi799/health-wellness-mlops/pull/42) — Grafana dashboard and Docker Compose stack (`p4-grafana-dashboard`)**:
   - Configured the 4-service stack in `docker-compose.yml`:
     - `init` (`python:3.11-slim`, root `0:0`): creates `/app/logs` on `./logs` and sets ownership `1000:1000` and mode `775`.
     - `api` (built from `Dockerfile`, port `8000:8000`, non-root `appuser`): waits for `init` (`service_completed_successfully`), mounts `./logs:/app/logs`, and exposes `/health` healthcheck.
     - `prometheus` (`prom/prometheus:v3.4.1`, port `9090:9090`): waits for `api` (`service_healthy`) and scrapes `api:8000/metrics` every `5s` using read-only `monitoring/prometheus/prometheus.yml`.
     - `grafana` (`grafana/grafana:11.6.0`, port `3000:3000`): depends on `prometheus`, persists state in `grafana-data`, sets `GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}` without a fallback default, and auto-provisions datasource and dashboard files from `monitoring/grafana/`.
   - Authored the seven-panel Grafana dashboard (`monitoring/grafana/dashboards/health-wellness-api.json`, `uid: health-wellness-api`):
     1. *Requests per minute* (`timeseries` via `api_requests_total`)
     2. *4xx/5xx error rate* (`timeseries` percentage over `5m`)
     3. *Request latency p50 and p95* (`timeseries` quantiles via `api_request_latency_seconds_bucket`)
     4. *Predictions by risk band* (`timeseries` stacked bars via `predictions_total`)
     5. *Average risk score over time* (`timeseries` via `prediction_risk_score_sum / prediction_risk_score_count`)
     6. *Risk score histogram* (`heatmap` via `prediction_risk_score_bucket`)
     7. *Serving model version* (`table` instant query via `model_info`)

### Verification: tested behavior vs. runtime traffic simulation

- **Automated test verification (`127 passed, 3 warnings`)**: Running `pytest tests/test_api_monitoring.py tests/test_api_metrics.py tests/test_monitoring_stack.py tests/test_simulate_traffic.py` yields **127 passed, 3 warnings** (third-party `StarletteDeprecationWarning`s from `TestClient`/`httpx`). These tests verify:
  - `PredictionLogger` receives all 21 inputs (including `null` optionals), writes JSONL on `HTTP 200`, is not called on `HTTP 422`, and swallows filesystem exceptions without breaking responses (`test_api_monitoring.py`, `test_api_metrics.py`).
  - `X-Request-ID` validation/echoing, UUID v4 fallback, response-body invariance, and absence of patient inputs in `health_api` logs (`test_api_monitoring.py`).
  - Exact six Prometheus metric schemas, bucket boundaries, custom registry isolation, `GET /metrics` scrape exclusion, bounded route templates, status code accounting (`200`, `405`, `422`, `500`, `503`), and zero PII/request IDs in `/metrics` (`test_api_metrics.py`).
  - Static validation of `docker-compose.yml`, `prometheus.yml`, Grafana provisioning YAML, all seven dashboard panels/PromQL queries, and secret hygiene (`test_monitoring_stack.py`).
  - Traffic simulator feature filtering, rate pacing, remote-URL limits, and `TestClient` status counting (`test_simulate_traffic.py`).
- **Runtime traffic simulation output**: Manually running `python -m scripts.simulate_traffic --batch data/processed/test.csv --rate 5 --limit 20 --bad 5` against the local API returned **20 HTTP 200, 5 HTTP 422, 0 other errors, 0 network errors**. (Individual per-line inspection of all 20 generated JSONL records from that replay and Grafana UI screenshots from the replay are not stored as separate verification artifacts.)
- **Runbook documentation**: Operational setup, commands, troubleshooting, and an end-to-end demo procedure are documented in the [monitoring runbook](monitoring-runbook.md).

### Implemented features vs. limitations and future work

- **Implemented features**:
  - Request-ID lifecycle tracing (`X-Request-ID` allowlist validation and UUID v4 generation).
  - Synchronous, thread-safe (in-process), lazy-initialized, error-isolated JSONL prediction logging to `logs/predictions.jsonl`.
  - Six Prometheus application metrics on `/metrics` with bounded route labels and `/metrics` scrape exclusion.
  - Four-service Docker Compose stack (`init`, `api`, `prometheus`, `grafana`) with healthchecks, non-root execution, and auto-provisioned 7-panel Grafana dashboard.
- **Limitations & future work**:
  - *Render ephemeral filesystem*: The public service on Render Free tier runs on an ephemeral disk that is cleared on restarts, deploys, and idle sleep; local JSONL logs inside Render do not persist, and Prometheus/Grafana are **not** deployed on Render.
  - *Cross-process log locking*: `PredictionLogger.file_lock` synchronizes threads within a single worker process (matching the default single-worker Uvicorn container command), not across multiple worker processes.
  - *Cloud persistence and alerting*: Future production enhancements include forwarding prediction logs to durable cloud object storage or log pipelines, hosting managed Prometheus/Grafana, and wiring Alertmanager rules for error-rate or drift notifications.
