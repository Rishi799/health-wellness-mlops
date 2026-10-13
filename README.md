# AI-Assisted Health and Wellness Awareness Platform

An end-to-end MLOps project that estimates a person's lifestyle-related health risk (0–100 score, Low / Moderate / High) from basic health and lifestyle data, and demonstrates the full ML lifecycle: data validation, versioning, experiment tracking, deployment, monitoring, drift detection and retraining.

> **Disclaimer:** awareness and educational tool only — not a medical diagnosis.

| | |
| --- | --- |
| Course | AI252TA Machine Learning Operations — Topic 27 |
| SDGs | 3 Good Health and Well-being · 4 Quality Education |
| Team | Person 1 · Person 2 · Person 3 · Person 4 |
| Phase | 1 — Development and reproducibility |

## Dataset
NHANES (CDC, public domain, no personal identifiers). 2017–2018 (9,254 people) for training; 2021–2023 (11,933 people) kept for Phase 3 drift. 15 files per cycle, joined on `SEQN`. See `DATA_README.md` and `DATA_DICTIONARY.xlsx`.

## Tech stack
Python 3.11 · pandas · scikit-learn · pandera · Jupyter · Git/GitHub · DVC · MLflow — later FastAPI, Docker, GitHub Actions, Prometheus/Grafana, Evidently.

## Setup
```bash
git clone https://github.com/Rishi799/health-wellness-mlops.git
cd health-wellness-mlops
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python download_nhanes.py
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
docs/              role handbooks
```

## Team workflow
`main` is protected: work on a branch ('p1-Rishi', 'p2- Rishabh'  , 'p3-Rhea' , 'p4-John'), open a pull request, one teammate reviews. Pin new packages in `requirements.txt`. Never edit raw data.

## Data source
CDC / NCHS, National Health and Nutrition Examination Survey. https://wwwn.cdc.gov/nchs/nhanes/