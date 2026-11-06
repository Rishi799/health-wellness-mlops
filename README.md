# AI-Assisted Health and Wellness Awareness Platform

[![CI](https://github.com/Rishi799/health-wellness-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Rishi799/health-wellness-mlops/actions/workflows/ci.yml)
[![CD](https://github.com/Rishi799/health-wellness-mlops/actions/workflows/cd.yml/badge.svg)](https://github.com/Rishi799/health-wellness-mlops/actions/workflows/cd.yml)

An end-to-end MLOps project that estimates a person's cardiometabolic risk (0–100 score, Low / Moderate / High) from basic health and lifestyle data, and demonstrates the ML lifecycle: data validation, versioning, experiment tracking, model registry, testing, containerisation, CI/CD and deployment (monitoring and drift detection follow in Phase 3).

> **Disclaimer:** awareness and educational tool only — not a medical diagnosis.

**Live API:** https://health-wellness-api.onrender.com — interactive docs at [`/docs`](https://health-wellness-api.onrender.com/docs), health at [`/health`](https://health-wellness-api.onrender.com/health). The free instance sleeps after about 15 minutes idle; the first request can take up to a minute.

| | |
| --- | --- |
| Course | AI252TA Machine Learning Operations — Topic 27 |
| SDGs | 3 Good Health and Well-being · 4 Quality Education |
| Team | Rishi Agarwal · Rishabh · Rhea · Niranjan (John) |
| Phase | 2 — Model deployment and productionisation |

## Dataset
NHANES (CDC, public domain, no personal identifiers). 2017–2018 (9,254 people) for training; 2021–2023 (11,933 people) kept for Phase 3 drift. 15 files per cycle, joined on `SEQN`. See `DATA_README.md` and `DATA_DICTIONARY.xlsx`.

## Prediction target
`cardiometabolic = 1` if an adult (20+) was ever told by a doctor they have **diabetes (borderline counts as yes), heart failure, coronary heart disease, angina, a heart attack or a stroke**; `0` if all six answers are no. Adults whose answers leave the target unknown (only "don't know" instead of "no") are dropped.

| 2017–2018 | Value |
| --- | --- |
| Adults in the dataset | 5,536 |
| At risk (`cardiometabolic = 1`) | 1,444 (26.1%) |
| Train / test (stratified 80/20, seed 42) | 4,428 / 1,108 — 26.1% at risk in each |

The target is imbalanced (about 1 in 4), so models are judged on ROC-AUC and recall, not accuracy.

## Data pipeline (Phase 1)
Every step is a DVC stage in `dvc.yaml`; `dvc repro` rebuilds everything from the raw files.

```
raw NHANES ─► people · body · lifestyle · mental ─► merge ─► validate ─► split ─► train.csv / test.csv
```

| Stage | Script | Output |
| --- | --- | --- |
| people_2017 | `src/data/people.py` | adults 20+, target |
| body_2017 | `src/data/body.py` | body measures, blood pressure, labs |
| lifestyle_2017 | `src/data/lifestyle.py` | smoking, alcohol, activity, sleep, diet |
| mental_2017 | `src/data/mental.py` | PHQ-9 score and category |
| merge_2017 | `src/data/merge.py` | one row per adult, 31 columns |
| validate_2017 | `src/data/validate.py` | pandera checks on every table (358 checks); stops the pipeline on failure |
| split | `src/data/split.py` | `data/processed/train.csv`, `test.csv` (settings in `params.yaml`) |

Columns that must **not** be used as model features (data leakage): `seqn`, the six target parts (`diabetes`, `heart_failure`, `coronary_heart_disease`, `angina`, `heart_attack`, `stroke`), `hba1c` (HbA1c ≥ 6.5% is the diagnosis of diabetes) and `diet_quality` (not available in 2021–2023).

## Phase 2: from model to service

```
compare_models.py ─► MLflow Model Registry ─► export_model.py ─► Docker image ─► Render
 (7 models)           cardiometabolic-risk-model   models/model.skops    (GHCR)        public API
                      alias champion → v2          (tracked by DVC)
```

### Model comparison
Seven models on the same test set (1,108 adults), tracked in MLflow on DagsHub. Logistic Regression is the champion.

| Model | ROC-AUC | Recall | Precision | F1 |
| --- | --- | --- | --- | --- |
| **Logistic Regression** | **0.850** | **0.827** | 0.549 | **0.660** |
| Gradient Boosting | 0.840 | 0.502 | 0.650 | 0.566 |
| Random Forest | 0.840 | 0.616 | 0.599 | 0.608 |
| Extra Trees | 0.836 | 0.367 | 0.663 | 0.472 |
| XGBoost | 0.824 | 0.702 | 0.529 | 0.603 |
| Decision Tree | 0.725 | 0.685 | 0.490 | 0.571 |
| Dummy (baseline) | 0.500 | 0.000 | 0.000 | 0.000 |

### Inference API (FastAPI)
| Endpoint | Purpose |
| --- | --- |
| `GET /health` | service and model status (503 if the model is not usable) |
| `GET /model-info` | model name, version, features, metrics, `service_git_commit` |
| `POST /predict` | 21 features → probability, `risk_score` (0–100), `risk_band`, disclaimer |

Only `age` (20–80) and `sex` (1 = male, 2 = female) are required; missing optional features are imputed by the model pipeline. Invalid input returns HTTP 422. Risk bands: **Low < 30 · Moderate 30–59 · High ≥ 60**.

```bash
curl -X POST https://health-wellness-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 55, "sex": 1, "bmi": 30.0, "sbp": 140.0, "told_high_bp": 1.0}'
```
```json
{"probability": 0.5965, "risk_score": 59.6, "risk_band": "Moderate", "model_version": "2", "disclaimer": "..."}
```
More examples: [`docs/api_examples.md`](docs/api_examples.md).

### Model quality
`python -m src.models.quality_gate` evaluates the exported champion and exits with code 1 if a hard rule fails, so CI blocks the merge. Report: `reports/model_quality.json`; details in the [model card](docs/model_card.md).

| Rule | Limit | Value | Result |
| --- | --- | --- | --- |
| ROC-AUC | ≥ 0.80 | 0.850 | PASS |
| Recall (threshold 0.5) | ≥ 0.75 | 0.827 | PASS |
| Improvement over Dummy | ≥ 0.20 | 0.350 | PASS |
| No leakage · input contract · artifact features | true | true | PASS |
| Brier score (warning) | ≤ 0.20 | 0.162 | PASS |
| Subgroup recall (warning) | ≥ 0.60 | 0.118 for ages 20–39 | WARN |

Known limitation: only 2 of 17 at-risk adults aged 20–39 are detected; predicted probabilities are over-estimated at the high end. A low score must not be used to dismiss symptoms.

### CI/CD (GitHub Actions)
| Workflow | When | What it does |
| --- | --- | --- |
| CI — Offline tests (no secrets) | every PR and push | tests that need no data or credentials |
| CI — Data checks and tests *(required)* | every PR and push | `dvc pull`, rebuild and verify the locked pipeline, all tests, model quality gate, upload quality report |
| CI — Docker image builds and starts *(required)* | every PR and push | build the image, run it, smoke-test `/health` and `/predict`, check non-root and read-only model |
| CD | after CI succeeds on `main` | build and verify the image, publish `ghcr.io/rishi799/health-wellness-api:<commit>` and `:latest`, deploy that commit to Render and verify the live service reports it |

`main` is protected: a pull request needs one approval and both required checks. Every merge is deployed automatically; previous images remain tagged by commit for rollback ([runbook](docs/render-deployment.md)).

## Tech stack
Python 3.11 · pandas · scikit-learn · XGBoost · pandera · pytest · FastAPI · pydantic · skops · Docker · GitHub Actions · GitHub Container Registry · Render · Git/GitHub · DVC (remote on DagsHub) · MLflow (tracking and Model Registry on DagsHub) — Phase 3: Prometheus/Grafana, Evidently.

## Setup & Quickstart

### 1. Local development
```bash
git clone https://github.com/Rishi799/health-wellness-mlops.git
cd health-wellness-mlops
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Get the data and rebuild the pipeline (DagsHub credentials are set once with `dvc remote modify origin --local ...` and are never committed):
```bash
dvc pull          # data + models/model.skops (or: python download_nhanes.py for raw data only)
dvc repro         # runs all stages; "Data and pipelines are up to date" when nothing changed
dvc dag           # shows the pipeline graph
pytest            # automated tests (pytest --offline needs no data or credentials)
python -m src.models.quality_gate   # model quality gate
```

### 2. Run the API locally
```bash
uvicorn src.api.main:app --reload      # http://localhost:8000/docs
```

### 3. Run the API with Docker
```bash
dvc pull models/model.skops.dvc
docker compose up --build              # http://localhost:8000/docs
curl http://localhost:8000/health
```
See [`docs/docker.md`](docs/docker.md) and the [Render deployment and rollback runbook](docs/render-deployment.md).

## Project structure
```
.github/workflows/ ci.yml (3 jobs) · cd.yml (publish + deploy)
data/raw/nhanes/   original CDC files (never edited, tracked by DVC)
data/interim/      topic tables + merged table
data/processed/    train.csv, test.csv
models/            champion model artifact (.skops via DVC) & metadata
notebooks/         exploration notebooks
src/data/          data scripts
src/models/        comparison, registration, export & quality-gate scripts
src/api/           FastAPI inference service & validation schemas
scripts/           smoke test and Render deploy helpers
tests/             unit, model, API, contract and deployment tests
reports/           figures, validation and model-quality reports
docs/              decision logs, data dictionaries, model card, API examples, Docker & Render runbooks
dvc.yaml           pipeline stages · params.yaml  pipeline settings
Dockerfile         slim production container image (python:3.11-slim, non-root)
docker-compose.yml local container execution on port 8000
```

## Key findings
- Risk rises steeply with age: 4.7% (20–39) → 21.5% (40–59) → 45.3% (60–79) → 52.0% (80+).
- Men are at higher risk than women: 29.0% vs 23.4%.
- Diabetes dominates the target (18.9% of adults); each heart condition or stroke is 2.9–4.9%.
- HbA1c leaks the target and is excluded from the model features.
- The strongest predictors of the champion model are age, a prior high-blood-pressure diagnosis, PHQ-9 score, pulse and waist.

Charts: `reports/figures/`.

## Team
| Member | Phase 1 | Phase 2 |
| --- | --- | --- |
| Rishi Agarwal | people table, target, merge, split, repository | CI, quality-gate and Docker checks, CD, deployment, release |
| Rishabh | body and clinical data, DVC pipeline | Docker, line-ending fix, deployment hardening, Render |
| Rhea | lifestyle data, data validation | model tests, quality gate, fairness checks, model card |
| Niranjan (John) | mental-health data, MLflow, baseline models | 7-model comparison, Model Registry, export, FastAPI service |

## Team workflow
`main` is protected: work on a branch, open a pull request, one teammate reviews, and both required CI checks must pass. Pin new packages in `requirements.txt` (runtime packages also in `requirements-api.txt`). Never edit raw data; never commit data, models, `.env` or tokens.

## Data source
CDC / NCHS, National Health and Nutrition Examination Survey. https://wwwn.cdc.gov/nchs/nhanes/
