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
- **Diastolic Auscultation Artifacts:** 134 readings in `BPXDI1` had 0 mmHg diastolic pressure. Treating 0 mmHg as a valid physiological reading would severely bias blood pressure averages downward.

### 3. How it was fixed
- **Dependency Pin:** Replaced `soupsieve==3.0.3` with `soupsieve==2.10`, which fully supports Python 3.11.0, is compatible with `beautifulsoup4==4.15.0`, and allowed all 254 pinned requirements to resolve cleanly.
- **Near-Zero Snapping:** Implemented `read_xpt` with `df.mask(df.abs() < 1e-10, 0.0)` to normalize SAS float artifacts to true `0.0`.
- **Clinical BP Cleaning:** Any diastolic reading `<= 0` mmHg is converted to `NaN` prior to computing the arithmetic mean across valid seated readings.
- **Target Leakage Protocol:** Documented that $\text{HbA}_{1\text{c}} \ge 6.5\%$ defines diabetes. Since diabetes is part of the prediction target, `hba1c` must be excluded from feature training.
