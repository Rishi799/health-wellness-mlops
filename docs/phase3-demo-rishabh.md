# Rishabh’s Phase 3 demo

Allow about four minutes for this segment of the team presentation. Start the Compose stack before the presentation so image pulls and the API health check are out of the way. Have the [monitoring runbook](monitoring-runbook.md), [batch results](../reports/drift/batch_results.json), [degradation table](phase3-verification.md#degradation-experiment-the-unchanged-champion) and [trigger evidence](../reports/phase3/screenshots/retrain-requested.jpg) open.

## Preflight

- Verify Docker is running and the DVC artifacts are available. Set `GRAFANA_ADMIN_PASSWORD` in your shell or ignored `.env` as described in the monitoring runbook.
- Finish any tests first. Use a fresh prediction log for the rehearsal because both API tests and earlier traffic can populate `logs/predictions.jsonl`. Stop the local API before archiving an existing log, then restart it so it opens the fresh path. Preserve archived logs according to the project’s handling rules.
- Start the stack from the repository root and check the API, Prometheus, and Grafana.

```powershell
docker compose up -d --build --wait
docker compose ps
Invoke-RestMethod http://localhost:8000/health
```

- Open Grafana at `http://localhost:3000` and bring up **Health & Wellness API**. Use the dashboard as a handoff point to Niranjan’s segment. Allow a scrape interval for traffic to show up.
- Keep the [saved live-loop screenshot](../reports/drift/live_loop/evidently-live-loop.jpg), the [recorded drift summary](../reports/drift/live_loop/drift_summary.json) and the [recorded run verification](../reports/drift/live_loop/verification.json) handy. A complete 1,298-row run can be prepared ahead. The shorter live replay below demonstrates the request and logging path within the slot.

## Four-minute script

| Time | Show and say |
| --- | --- |
| 0:00–0:40 | Show the production batch folder. “I used the later NHANES survey cycle to model traffic after deployment. It gave us 7,783 validated adult records with 1,936 positive labels. I made six reproducible batches and a separate synthetic stress batch with older ages and shifted BMI and systolic blood pressure.” |
| 0:40–1:30 | Run the bounded replay below. “The simulator sends only the model’s serving inputs. Every 200 response writes a JSONL record with the model version, request ID and input fields. These five deliberate 422s demonstrate validation. I’ll show the record count rather than display individual health inputs.” Ask Niranjan to point out the request and prediction panels as metrics are scraped. |
| 1:30–2:35 | Run the drift command on `logs/predictions.jsonl` and open the saved full-batch report while it runs. “Evidently compares the logged features with the original training distribution. I use normalized Wasserstein distance for numeric features and Jensen–Shannon distance for categories. The complete ordinary batch had 10 of 21 features drift, and the synthetic stress batch had 13 of 21. Prediction drift is shown separately. The smaller live sample can produce different counts.” |
| 2:35–3:20 | Show the degradation table. “On the labelled later cohort, champion AUC fell from 0.8502 to 0.8248 and recall from 0.8270 to 0.7846. These metrics come from labels in the survey. The live JSONL shows inputs and scores, so it can answer the distribution question as soon as requests arrive.” |
| 3:20–4:00 | Run the trigger below and point to the reason. “Any one of three conditions requests retraining. Input drift reaches at least 30%, recall falls below 0.777, or AUC falls below 0.80. This matched, labelled stress run fires for input drift and the AUC floor. The identical-data control shows the opposite outcome. Rhea will show how we review the resulting challenger.” |

Run this during the demo from an activated environment with a patched Python 3.11. The shorter replay and trigger write to separate directories so the full run stays available.

```powershell
python -m scripts.simulate_traffic --api-url http://localhost:8000 --batch data/production/batches/batch_01.csv --limit 60 --rate 10 --bad 5
(Get-Content logs/predictions.jsonl | Measure-Object -Line).Lines
python -m src.monitoring.drift --current logs/predictions.jsonl --out reports/drift/live_demo
Start-Process (Resolve-Path 'reports/drift/live_demo/predictions_drift.html').Path
python -m src.monitoring.retrain_trigger --drift reports/drift/batch_shifted/drift_summary.json --performance reports/drift/batch_shifted/performance.json --out reports/drift/live_demo/stress_trigger.json
```

For the recorded full-run procedure, omit `--limit`, use `--rate 25`, and set the drift output to `reports/drift/live_loop`. Begin with a fresh log to reproduce the recorded row counts. The API log and the labelled CSV have different provenance, so use the existing matched labelled reports for the trigger demonstration.

## Faculty questions

| Question | Answer |
| --- | --- |
| Why did an ordinary batch drift? | All six later-cycle batches cross the fixed rule. They come from a later cohort with differences in measurement, survey questions and missingness. The identity control validates the zero-change case. |
| What does prediction drift measure here? | It compares the distribution of probabilities scored by the same champion from the original training inputs and the input fields captured in the API log. Evidently keeps that distribution separate from the 21-feature drift count. |
| Can the JSONL show a drop in recall? | Recall needs outcome labels. The log supplies serving inputs and prediction outputs, while the later survey cohort supplies labels for the degradation experiment. |
| Does a positive trigger deploy a new model? | It records the reasons to evaluate a candidate. The team reviews both holdouts and subgroup results before changing the champion. The later holdout detected only 3 of 14 positive young-adult cases with the challenger. |
| What if the demo stack is unavailable? | Show the saved Evidently screenshot, the full-run JSON verification, the labelled stress decision and the identity control. Identify them as evidence from the recorded run. |
