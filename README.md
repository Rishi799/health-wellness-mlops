# AI-Assisted Health and Wellness Awareness Platform

An end-to-end MLOps project that estimates a person's lifestyle-related health risk (0–100 score, Low / Moderate / High) from basic health and lifestyle data, and demonstrates the full ML lifecycle: data validation, versioning, experiment tracking, deployment, monitoring, drift detection and retraining.

> **Disclaimer:** awareness and educational tool only — not a medical diagnosis.

| | |
| --- | --- |
| Course | AI252TA Machine Learning Operations — Topic 27 |
| SDGs | 3 Good Health and Well-being · 4 Quality Education |
| Team | Rishi · Rishabh · Rhea · John |
| Phase | 1 — Development and reproducibility |

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

## Data pipeline
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
| validate_2017 | `src/data/validate.py` | pandera checks on every table; stops the pipeline on failure |
| split | `src/data/split.py` | `data/processed/train.csv`, `test.csv` (settings in `params.yaml`) |

Columns that must **not** be used as model features (data leakage): `seqn`, the six target parts (`diabetes`, `heart_failure`, `coronary_heart_disease`, `angina`, `heart_attack`, `stroke`), `hba1c` (HbA1c ≥ 6.5% is the diagnosis of diabetes) and `diet_quality` (not available in 2021–2023).

## Tech stack
Python 3.11 · pandas · scikit-learn · pandera · Jupyter · Git/GitHub · DVC (remote on DagsHub) · MLflow (on DagsHub) — later FastAPI, Docker, GitHub Actions, Prometheus/Grafana, Evidently.

## Setup
```bash
git clone https://github.com/Rishi799/health-wellness-mlops.git
cd health-wellness-mlops
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Get the data and rebuild the pipeline (DagsHub credentials are set once with `dvc remote modify origin --local ...` and are never committed):
```bash
dvc pull          # or: python download_nhanes.py
dvc repro         # runs all stages; "Data and pipelines are up to date" when nothing changed
dvc dag           # shows the pipeline graph
pytest            # unit tests
```

## Project structure
```
data/raw/nhanes/   original CDC files (never edited, tracked by DVC)
data/interim/      topic tables + merged table
data/processed/    train.csv, test.csv
notebooks/         exploration notebooks
src/data/          data scripts
src/models/        training scripts
reports/figures/   charts
docs/              decision logs, data dictionaries
dvc.yaml           pipeline stages · params.yaml  pipeline settings
```

## Key findings so far
- Risk rises steeply with age: 4.7% (20–39) → 21.5% (40–59) → 45.3% (60–79) → 52.0% (80+).
- Men are at higher risk than women: 29.0% vs 23.4%.
- Diabetes dominates the target (18.9% of adults); each heart condition or stroke is 2.9–4.9%.
- HbA1c leaks the target and is excluded from the model features.

Charts: `reports/figures/`.

## Team workflow
`main` is protected: work on a branch ('p1-Rishi', 'p2- Rishabh'  , 'p3-Rhea' , 'p4-John'), open a pull request, one teammate reviews. Pin new packages in `requirements.txt`. Never edit raw data.

## Data source
CDC / NCHS, National Health and Nutrition Examination Survey. https://wwwn.cdc.gov/nchs/nhanes/
