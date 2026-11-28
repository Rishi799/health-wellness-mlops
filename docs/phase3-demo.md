# Phase 3 team demo — run sheet

About **17 minutes** plus questions. Each segment has its own detailed script; this
page fixes the order, the hand-offs, the preparation and the fallback plan.

| Order | Time | Speaker | Segment | Detailed script |
| --- | --- | --- | --- | --- |
| 1 | 2 min | Rishi | Opening: goal, architecture, what changed since Phase 2 | below |
| 2 | 3.5 min | Niranjan | Observability: Compose stack, prediction log, `/metrics`, Grafana | [phase3-demo-niranjan.md](phase3-demo-niranjan.md) |
| 3 | 4 min | Rishabh | Production data, traffic replay, Evidently drift, degradation, trigger | [phase3-demo-rishabh.md](phase3-demo-rishabh.md) |
| 4 | 3.5–4 min | Rhea | Challenger vs champion, release decision, trigger sensitivity, governance | [phase3-demo-rhea.md](phase3-demo-rhea.md) |
| 5 | 2 min | Rishi | CI feedback loop, overall results, limitations, close | below |

**Hand-off lines to adjust:** Niranjan opens with "Thanks Rishi" (not "Thanks
Rishabh") and ends with "Over to you, Rishabh". Rishabh's and Rhea's scripts
already hand over in this order.

## Before the demo (30 minutes earlier)

- **Laptop with Docker (Rishabh or Niranjan):** `git pull` → `dvc pull` →
  `docker compose up -d --build --wait`; check `docker compose ps`.
- **Warm-up traffic** so Grafana already shows data and the error-rate spike:
  `python -m scripts.simulate_traffic --api-url http://localhost:8000 --batch data/production/batches/batch_01.csv --limit 200 --rate 10 --bad 5`
- **Open tabs:** Grafana `http://localhost:3000` (dashboard *Health & Wellness
  API*), Prometheus targets `http://localhost:9090/targets`, the live API
  `https://health-wellness-api.onrender.com/health` (wakes the free server),
  GitHub Actions (latest green CI run with the `monitoring-report` artifact),
  DagsHub MLflow models page (v2 `champion`, v3 `challenger`).
- **Slides:** `MediSentinel_Presentation.pptx` (16 slides) as the backbone.
- Close chats and any window showing `.env`, tokens or `.dvc/config.local`.

## 1. Opening — Rishi (2 min)

Slides 1, 5 and 8.

> "In Phase 2 we put our champion model — Logistic Regression, version 2 — into
> production on Render, with tests, a quality gate and CI/CD. Phase 3 answers
> the next question: once the model is live, how do we know it is still good,
> and what do we do when it is not?
>
> We added four things around the unchanged model: we **observe** every
> prediction and the service metrics; we **detect** drift and performance loss
> using the newer NHANES 2021–23 survey as production data; we **decide** with a
> fixed retraining trigger and a challenger model; and we **govern** the release
> with a human decision, a model card and fairness checks.
>
> Niranjan will start with how we observe the live service."

## 2–4. Team segments

Follow the three linked scripts. Keep to time; if a live step stalls for more
than 20 seconds, switch to that segment's saved screenshots.

## 5. Close — Rishi (2 min)

Slides 10, 13 and 15, then GitHub Actions.

> "Monitoring is also automated. Since PR #40, every CI build runs the drift and
> trigger check on a production batch and attaches the report — here is the
> `monitoring-report` artifact on the latest green run. A 'retrain needed'
> result does not fail the build; it is a decision for the team.
>
> Overall: our champion still holds ROC-AUC 0.825 on the later population, above
> our 0.80 floor, though recall fell from 0.827 to 0.785 and 8 of 21 inputs
> drifted. The trigger fired, a challenger was trained and compared, and we kept
> version 2 because the challenger loses about two points of recall on the
> original population.
>
> We report our limits openly: young adults are still poorly detected, labels
> are self-reported, the 30% drift rule is too sensitive at this batch size, and
> Grafana runs locally. Thank you — we are happy to take questions."

## Fallback plan

| Problem | Do this |
| --- | --- |
| Docker stack will not start | Use the saved Grafana screenshots in `reports/phase3/screenshots/` and `reports/drift/live_loop/` |
| Live Render API asleep | Say "the free server is waking up"; continue with the local stack |
| A command fails | Show the saved JSON/HTML evidence named in that segment's script |
| Question outside your part | Hand to the owner: data and drift → Rishabh, logging and dashboard → Niranjan, model review and governance → Rhea, CI and integration → Rishi |
