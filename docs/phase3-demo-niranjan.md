# Niranjan’s Phase 3 demo

Allow about three to four minutes for this segment of the team presentation. Start the Docker Compose stack beforehand so container builds and API health checks are complete. Have the [monitoring runbook](monitoring-runbook.md), Grafana dashboard (`http://localhost:3000`), Prometheus targets (`http://localhost:9090/targets`), and the four dashboard screenshots in [`reports/phase3/screenshots/`](../reports/phase3/screenshots/) ready before presenting.

This demonstration covers the **P4 monitoring contribution**: connecting live FastAPI inference with request-ID tracing, synchronous structured JSONL prediction logging, Prometheus metrics exposition, and the auto-refreshing seven-panel Grafana dashboard. It picks up immediately following Rishabh’s traffic replay and hands over to Rhea’s model governance and challenger review.

---

## Preflight checklist

- Verify Docker is running. Ensure `GRAFANA_ADMIN_PASSWORD` is set in your shell or gitignored `.env` file (with no fallback default, as required by `docker-compose.yml`).
- Start the four-service Compose stack from the repository root:
  ```bash
  docker compose up -d --build --wait
  docker compose ps
  ```
- Verify the API is healthy and model version 2 is loaded:
  ```bash
  curl -s http://localhost:8000/health
  ```
- Open Grafana at `http://localhost:3000` (sign in as `admin`), navigate to the **Health & Wellness API** dashboard (`uid: health-wellness-api`), set the time window to **Last 5 minutes** (or **Last 15 minutes**), and confirm auto-refresh is set to **5s**.
- Open Prometheus at `http://localhost:9090/targets` to confirm the `health-wellness-api` scrape target (`http://api:8000/metrics`) is `UP`.
- Keep a terminal split open to the repository root for running live inspection commands.
- Keep the four backup screenshots in [`reports/phase3/screenshots/`](../reports/phase3/screenshots/) open in an image viewer or browser tabs as an offline fallback.

---

## 3–4 minute demonstration script

| Time | Screen / Terminal Action | Spoken Words ("What I Say") | What the Audience Observes |
| --- | --- | --- | --- |
| **0:00–0:35**<br>*(35 sec)* | **Show presentation slide or Grafana dashboard header.** Point to the team handoff from Rishabh. | “Thanks Rishabh. In traditional software, monitoring checks whether a server is up or down. But in Machine Learning applications, models fail silently. An API can return HTTP 200 responses with low latency while making predictions on drifted data or generating anomalous risk scores.<br><br>My role in Phase 3 was implementing end-to-end monitoring and observability: capturing every inference event, exposing real-time operational and prediction metrics, and visualizing them in Grafana so we can detect degradation immediately.” | Title slide or Grafana header showing **Health & Wellness API**. Audience understands the motivation: ML systems require tracking both operational health and prediction distributions. |
| **0:35–1:10**<br>*(35 sec)* | **Show architecture diagram or terminal running `docker compose ps`.** Point out the four services. | “Here is our monitoring architecture. When client traffic arrives at our FastAPI inference service on port 8000, our middleware attaches a validated `X-Request-ID` and measures latency.<br><br>Successful predictions are synchronously logged to `logs/predictions.jsonl` on a host bind-mount. Simultaneously, our `APIMetrics` collector records metrics in an isolated Prometheus registry. Prometheus scrapes the `/metrics` endpoint every 5 seconds, and Grafana queries Prometheus to render our dashboard.<br><br>The entire stack runs locally via four Docker Compose services: `init` to set directory permissions, `api` running as unprivileged non-root `appuser`, `prometheus`, and `grafana`.” | Terminal displays `docker compose ps` showing `init` exited 0, `api` healthy on port 8000, `prometheus` running on port 9090, and `grafana` running on port 3000. |
| **1:10–1:50**<br>*(40 sec)* | **Switch to terminal.** Run:<br>`tail -n 1 logs/predictions.jsonl \| jq .`<br>and show a quick request test. | “Next, prediction logging and request tracing. Every request passing through our API receives an `X-Request-ID`—either preserving a valid client header or generating a UUID v4.<br><br>When a prediction succeeds with HTTP 200, `PredictionLogger` synchronously appends a JSON line to `logs/predictions.jsonl`. Here is a recorded entry: it includes the UTC timestamp, request ID, model version 2, latency in milliseconds, all 21 input features under `inputs`, probability, risk score, and risk band.<br><br>Importantly, our stdout logs never output patient data, and file writes are fully error-isolated: if disk I/O ever fails, a warning is logged, but the API response is never broken. And invalid HTTP 422 requests never write to this log.” | Terminal shows a cleanly formatted JSON line with `request_id`, `model_version: "2"`, `latency_ms`, all 21 input features, `probability`, `risk_score`, and `risk_band`. No PII in stdout. |
| **1:50–2:20**<br>*(30 sec)* | **In terminal, query `/metrics`:**<br>`curl -s http://localhost:8000/metrics \| grep -E "^(api_requests_total\|predictions_total\|model_info)"` | “Now let’s look at Prometheus metrics exposition at `GET /metrics`. We implemented six custom application metrics using an isolated `CollectorRegistry`.<br><br>This includes `api_requests_total` with bounded endpoint and status labels, `api_request_latency_seconds` with 11 histogram buckets, `predictions_total` partitioned by risk band, risk score and probability histograms, and a `model_info` gauge.<br><br>To avoid inflating our request counts, `GET /metrics` scrapes themselves are strictly excluded from request counting and latency tracking.” | Terminal outputs clean Prometheus metrics text showing counter and gauge values, including `model_info{git_commit="unknown",model_name="cardiometabolic-risk-model",model_version="2"} 1.0`. |
| **2:20–3:10**<br>*(50 sec)* | **Switch to browser at Grafana (`http://localhost:3000`).** Hover across panels from top to bottom. Point out the top-right auto-refresh indicator. | “Now let’s examine our Grafana dashboard, which auto-refreshes every 5 seconds to match Prometheus’s 5-second scrape interval.<br><br>At the top, we track operational health: **Requests per minute** tracks endpoint throughput, while the **4xx/5xx error rate** shows a clear red spike from the five invalid requests Rishabh sent during the simulation. Below that, **Request latency p50 and p95** confirms typical median inference latency stays under 15 milliseconds.<br><br>In the middle, we track model behavior: **Predictions by risk band** stacks predictions into Low, Moderate, and High bars. **Average risk score** shows our moving average around 41, and the **Risk score histogram** heatmap visualizes score distribution across all buckets.<br><br>Finally, at the bottom, **Serving model version** confirms we are actively serving champion version 2.” | Browser displays all seven Grafana panels dynamically updating with 5s auto-refresh: throughput spikes, red error-rate line, latency curves, stacked green/yellow/red risk bars, orange avg score line, heatmap, and model version table. |
| **3:10–3:40**<br>*(30 sec)* | **Summarize key takeaways and hand off to Rhea.** | “To conclude, our monitoring bridges live inference to MLOps engineering. When error rates spike or score distributions drift, the dashboard shows it within seconds (no automated alerting is configured yet). We can take a request ID from the logs and trace it directly into `predictions.jsonl` to inspect the exact inputs—which is how Rishabh ran drift analysis.<br><br>And the dashboard confirms model version 2 is actively serving. Now Rhea will explain how we evaluate whether a new challenger model should replace it. Over to you, Rhea.” | Audience sees how the observability stack links live serving back to offline data analysis, governance, and retraining decisions. Smooth handoff to Rhea. |

---

## Live demonstration commands

Execute these commands from the repository root in Bash or PowerShell during the demo:

### 1. Preflight status check
```bash
# Verify the 4 Compose services are running
docker compose ps

# Confirm the API healthcheck passes and model version 2 is loaded
curl -s http://localhost:8000/health
```

### 2. Inspect the latest prediction log entry
```bash
# View the most recent formatted prediction log record
tail -n 1 logs/predictions.jsonl | jq .

# Verify total prediction count
wc -l logs/predictions.jsonl
```

### 3. Inspect Prometheus metrics
```bash
# Check application metrics exposed at /metrics
curl -s http://localhost:8000/metrics | grep -E "^(api_requests_total|predictions_total|model_info)"
```

### 4. Optional quick traffic burst (if additional live points are needed)
```bash
# Send 20 valid rows and 5 deliberate 422 validation errors
python -m scripts.simulate_traffic --batch data/processed/test.csv --rate 5 --limit 20 --bad 5
```

---

## Seven dashboard panels reference

The Grafana dashboard ([`monitoring/grafana/dashboards/health-wellness-api.json`](../monitoring/grafana/dashboards/health-wellness-api.json)) defines seven panels:

| Panel # | Panel Title | Visualization Type | Metric / PromQL Expression | Purpose |
|---|---|---|---|---|
| **1** | Requests per minute | `timeseries` | `sum by (endpoint) (rate(api_requests_total[1m])) * 60` | Tracks incoming HTTP throughput (req/min) partitioned by bounded endpoint (`/predict`, `/health`). |
| **2** | 4xx/5xx error rate | `timeseries` | `100 * sum(rate(api_requests_total{status=~"4..\|5.."}[5m])) / sum(rate(api_requests_total[5m]))` | Percentage of client/server errors over 5m. Shows fixed red line highlighting validation failures (`HTTP 422`). |
| **3** | Request latency p50 and p95 | `timeseries` | `histogram_quantile(0.50, sum by (le, endpoint) (rate(api_request_latency_seconds_bucket[5m])))`<br>`histogram_quantile(0.95, ...)` | Tracks 50th and 95th percentile latency per endpoint in seconds; reveals performance bottlenecks. |
| **4** | Predictions by risk band | `timeseries` (stacked bars) | `sum by (risk_band) (rate(predictions_total[5m]))` | Real-time prediction rate (`pred/s`) stacked by `Low` (green), `Moderate` (yellow), and `High` (red). |
| **5** | Average risk score over time | `timeseries` | `sum(rate(prediction_risk_score_sum[5m])) / sum(rate(prediction_risk_score_count[5m]))` | Moving average of predicted cardiometabolic risk scores (0–100, fixed orange line). |
| **6** | Risk score histogram | `heatmap` | `sum(increase(prediction_risk_score_bucket[5m])) by (le)` | Heatmap showing density distribution of predicted risk scores across 10-point buckets over time. |
| **7** | Serving model version | `table` (instant query) | `model_info` | Instant table displaying loaded champion metadata: `Model Name`, `Model Version` (`2`), and `Git Commit`. |

---

## Faculty evaluation questions & answers

| Question | Recommended Answer |
|---|---|
| **Why log predictions to a JSONL file rather than directly to a relational database?** | Writing append-only JSONL to disk provides high-throughput, low-latency logging without introducing an external database network dependency or schema migration bottlenecks during inference. The log serves as an immediate local sink that downstream tools (like Evidently drift analysis) consume directly. In a cloud architecture, this file would be streamed to object storage like S3 or GCS. |
| **What happens if the log directory is full or unwritable? Does inference fail?** | No. In `PredictionLogger.log_prediction()`, all file I/O is enclosed in `try ... except Exception` blocks. If an error occurs, it emits a warning to the `health_api` logger, but the API still returns the HTTP 200 response with risk scores to the user intact. Logging failures never degrade user-facing predictions. |
| **Why are `GET /metrics` scrapes excluded from HTTP request metrics?** | Prometheus scrapes `/metrics` every 5 seconds. If included, scrape traffic would dominate request counts and skew latency histograms toward sub-millisecond metric reads rather than reflecting actual inference workload on `/predict`. |
| **Why does the dashboard use a 5-second auto-refresh?** | Our Prometheus configuration in `prometheus.yml` is set to `scrape_interval: 5s`. Setting Grafana's refresh interval to 5 seconds synchronizes the visualization with new data arrivals, giving near-real-time feedback during traffic surges without excessive browser polling. |
| **How does request ID tracing protect patient privacy?** | The `request_id_and_latency_middleware` logs `method`, `path`, `status`, `latency_ms`, and `request_id` to stdout/stderr. Patient inputs (age, BMI, blood pressure) are omitted from standard application logs to prevent PII exposure in log aggregators. The full feature payload is written only to `predictions.jsonl`, tied to that unique request ID; in production that file would need access control and a retention period. |
| **Why does Render Free tier not run Prometheus and Grafana?** | Render Free tier uses an ephemeral filesystem that wipes local files upon restart or after 15 minutes of inactivity. Running our full four-container Compose stack locally allows persistent host storage for logs and persistent named volumes for Grafana. In production, Grafana and Prometheus would run as managed cloud services. |
| **How does `PredictionLogger` handle concurrency?** | It uses an internal `threading.Lock` around file append and flush operations. This ensures thread-safety across concurrent requests within the single-worker Uvicorn process used in our container deployment. |

---

## Backup plan (offline fallback)

If Docker, Prometheus, or Grafana is unavailable during the live presentation, do not troubleshoot live. Switch immediately to the pre-captured evidence:

1. **Overall Dashboard Overview:** Open [`reports/phase3/screenshots/01-overall-dashboard.png`](../reports/phase3/screenshots/01-overall-dashboard.png) to show the full seven-panel dashboard displaying simulated NHANES traffic, the 5-second refresh, and all operational and prediction metrics.
2. **Request Rate, Error Rate & Latency:** Open [`reports/phase3/screenshots/02-request-monitoring.png`](../reports/phase3/screenshots/02-request-monitoring.png) to show the throughput curves, the red error-rate spike from the 5 invalid 422 requests, and p50/p95 latency metrics.
3. **Prediction Risk Distribution:** Open [`reports/phase3/screenshots/03-prediction-monitoring.png`](../reports/phase3/screenshots/03-prediction-monitoring.png) to show stacked risk bands (Low/Moderate/High), the average risk score line (~41), and the score heatmap.
4. **Serving Model Metadata:** Open [`reports/phase3/screenshots/04-serving-model-version.png`](../reports/phase3/screenshots/04-serving-model-version.png) to show the table panel confirming model name `cardiometabolic-risk-model` and model version `2`.
5. **Logged Predictions Verification:** Show the existing test suite results (`pytest tests/test_api_monitoring.py tests/test_api_metrics.py tests/test_monitoring_stack.py`), which verify all 112 monitoring and dashboard assertions pass offline.
