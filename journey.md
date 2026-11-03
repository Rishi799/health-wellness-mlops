# Engineering Journey Log

A running engineering log of progress, challenges encountered, root-cause analyses, and resolutions across project phases.

---

## Week 1: Phase 1 Setup, DVC Initialization & Clinical Pipeline (Person 2 — Rishabh)

### 1. What was done
- Inspected the repository and authoritative NHANES raw datasets (`2017-2018` development cycle and `2021-2023` drift cycle).
- Configured Python 3.11 virtual environment and resolved package dependency compatibility.
- Initialized DVC repository (`dvc init`) and tracked 30 raw SAS transport files in `data/raw/nhanes` via DVC (`data/raw/nhanes.dvc`), keeping large binaries out of Git.
- Implemented `src/data/body.py` to process and clean anthropometrics (BMX), blood pressure (BPX/BPXO), questionnaire (BPQ), and laboratory results (TCHOL, HDL, GHB) into `data/interim/body_<cycle>.csv` under the team data contract.
- Built automated DVC stages (`dvc.yaml` & `dvc.lock`) tracking `people_2017` and `body_2017` with full dependency tracking and cache idempotency (`dvc repro` and `dvc dag`).
- Wrote automated unit tests in `tests/test_body.py` (7 tests, 100% pass rate).
- Documented clinical reasoning in `docs/data-cleaning-decisions.md` and schema definitions in `docs/person2-data-dictionary.md`.
- Generated 4 publication-quality exploratory figures in `reports/figures/` (including an explicit target leakage analysis chart for HbA1c).
- Created interactive exploration notebook `notebooks/p2_body.ipynb`.

### 2. What broke & challenges encountered
- **Python Version Conflict:** `soupsieve==3.0.3` pinned in `requirements.txt` failed to install on Python 3.11.0 with `No matching distribution found`. Investigation revealed that SoupSieve 3.0+ yanked/restricted support on Python `< 3.11.5` due to a CPython regex engine bug in possessive quantifiers.
- **SAS Float Encoding Quirk:** True zero values in raw SAS transport files are stored as near-zero floats (e.g., `5.397605e-79`). Simple equality checks (`x == 0`) failed to detect these values.
- **Diastolic Auscultation Artifacts & Outliers:** 134 readings in `BPXDI1` had 0 mmHg diastolic pressure, and 5 ambulatory readings had non-physiological DBP < 20 mmHg (e.g. 8, 12, 14.7 mmHg). Treating these as valid physiological readings severely biased blood pressure downward.
- **Validation Schema Failures (validate.py):** Automated Pandera schema checks caught 3 boundary issues: DBP < 20 mmHg, 2 extreme outlier BMI readings > 80 kg/m² (84.4, 86.2), and toddler weights < 10 kg in the unmerged survey cohort.

### 3. How it was fixed
- **Dependency Pin:** Replaced `soupsieve==3.0.3` with `soupsieve==2.10`, which fully supports Python 3.11.0, is compatible with `beautifulsoup4==4.15.0`, and allowed all 254 pinned requirements to resolve cleanly.
- **Near-Zero Snapping:** Implemented `read_xpt` with `df.mask(df.abs() < 1e-10, 0.0)` to normalize SAS float artifacts to true `0.0`.
- **Clinical BP & BMI Boundaries:** Upgraded diastolic filter to drop any reading `< 20.0` mmHg as an auscultation/cuff artifact, and bounded BMI to `[12.0, 80.0]` kg/m² setting extreme outliers to `NaN`.
- **Target Leakage Protocol:** Documented that $\text{HbA}_{1\text{c}} \ge 6.5\%$ defines diabetes. Since diabetes is part of the prediction target, `hba1c` must be excluded from feature training.
- **DagsHub Cloud Storage:** Configured DagsHub HTTPS remote with basic authentication isolated in `.dvc/config.local` (git-ignored), successfully uploading all 30 raw XPT files (37.8 MB) and pipeline interim outputs.

---

## Week 2: End-to-End DVC Pipeline, Multi-Topic Integration & Reproducibility (Person 2 — Rishabh)

### 1. What was done
- Integrated all 7 data stages into `dvc.yaml` (`people_2017`, `body_2017`, `lifestyle_2017`, `mental_2017`, `merge_2017`, `validate_2017`, and `split`).
- Established pipeline gatekeeper: `split` depends on `reports/validation_2017-2018.json`, guaranteeing that train/test sets are only created if Pandera contract validation passes with 0 errors.
- Verified deterministic execution with `dvc repro`:
  - `merged_2017-2018.csv`: 5,536 adults × 31 columns (26.1% at risk).
  - `validation_2017-2018.json`: all 5 tables PASSED (358 total checks, 0 failed).
  - `train.csv`: 4,428 rows (26.1% at risk); `test.csv`: 1,108 rows (26.1% at risk).
- Verified cache idempotency: second `dvc repro` run reported `Data and pipelines are up to date`.
- Pushed all interim datasets and processed splits (`train.csv`, `test.csv`) to DagsHub remote storage (`dvc push` -> cache and remote in sync).
- Generated full DAG graph (`dvc dag`) for Phase 1 review slides.


---

## Week 3: Phase 2 Docker Containerisation, Environment Management & Cross-Platform Reproducibility (Person 2 — Rishabh)

### 1. What was done
- **Production Containerisation:** Created `Dockerfile` based on `python:3.11-slim` with a non-root user (`appuser`), Docker layer caching (copying dependencies before application code), automatic `HEALTHCHECK` probing `/health`, and execution on port 8000.
- **Slim API Runtime Requirements:** Created `requirements-api.txt` pinned to only the 7 minimal dependencies required for inference (`fastapi`, `uvicorn`, `pydantic`, `pandas`, `numpy`, `scikit-learn`, `skops`), keeping container size under target (< 600 MB) rather than bundling the 254-package dev environment.
- **Local Composition & Operations Runbook:** Added `docker-compose.yml` for single-command developer onboarding (`docker compose up --build`) and documented container lifecycle, troubleshooting, and security in `docs/docker.md`.
- **Cross-Platform Hash Parity:** Solved Windows/Linux DVC hash divergence by authoring `.gitattributes` (`* text=auto eol=lf` plus binary exceptions for `.xpt`, `.png`, `.skops`, `.xlsx`, `.pptx`, `.pdf`) and enforcing `lineterminator="\n"` on all data pipeline writers in `src/data/*.py`.
- **Pipeline Re-synchronization:** Ran `dvc repro` to regenerate all 7 stages with deterministic LF line endings, validated data integrity with Pandera (358 checks passed, 0 errors), and pushed updated cache to DagsHub remote via `dvc push`.
- **Lockfile Integrity:** Recorded clean LF md5 hashes into `dvc.lock` so that GitHub Actions CI (`dvc status`) passes with `Data and pipelines are up to date`.
- **Main Branch Integration:** Merged `origin/main` containing Niranjan's model registry and FastAPI service (PR #15) and Rishi's CI workflow (PR #17), resolving placeholder API conflicts in favor of the production service. Verified 100% pass rate across the full 124-test suite (`pytest`).

### 2. What broke & challenges encountered
- **CRLF vs. LF Line-Ending Divergence:** Because Windows writes CRLF (`\r\n`) by default and Linux CI checks out LF (`\n`), code files (e.g. `src/data/people.py`) had different md5 hashes on developer machines versus GitHub Actions runners. This caused `dvc status` on Linux CI to falsely report that every pipeline stage was out of date.
- **API Placeholder Merge Conflict:** To allow Dockerfile authoring and smoke testing in parallel before the API was merged, a placeholder `src/api/main.py` was created. Merging `origin/main` caused an `add/add` conflict against Niranjan's production FastAPI application.
- **Root vs. Subfolder `.gitignore` Conflicts:** A root `.gitignore` entry `models/*` inadvertently shadowed Niranjan's tracking of `models/model.skops.dvc` and `models/model_info.json`.
- **Starlette TestClient Socket Restrictions:** When running integration tests under restricted Windows command sandboxes without loopback networking, Starlette's `TestClient` ASGI lifespan initialization blocked on socket setup.

### 3. How it was fixed
- **Git Line Normalization:** Configured `.gitattributes`, ran `git add --renormalize .`, and updated all `to_csv()` and `json.dump()` calls across `src/data/*.py` with explicit newline parameters (`lineterminator="\n"`).
- **Conflict Resolution Protocol:** Resolved API conflicts using `git checkout --theirs src/api/main.py src/api/__init__.py` to adopt the production application, removing obsolete placeholders and `.gitkeep` markers.
- **Scoped Artifact Tracking:** Cleaned root `.gitignore` to allow `models/` metadata tracking, relying on `models/.gitignore` specifically for the untracked binary `model.skops`.
- **Verified Local & Remote Pipeline:** Executed `dvc pull models/model.skops.dvc`, verified `dvc status` clean, and confirmed complete test suite passing (124 passed in ~14s).
