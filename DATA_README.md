# Raw data — NHANES (CDC National Health and Nutrition Examination Survey)

Public-use U.S. health survey data, downloaded unchanged from the CDC. `SEQN` is the survey participant linkage key; it is excluded from model features and API requests. The derived datasets are unweighted educational ML cohorts, not population prevalence estimates.

## Where the data is

| Path | Contents |
| --- | --- |
| `data/raw/nhanes/2017-2018/` | 15 original files (`.xpt`, SAS transport format), suffix `_J` — **training data**, 9,254 people |
| `data/raw/nhanes/2021-2023/` | 15 original files, suffix `_L` — **later population for Phase 3 drift**, 11,933 people |
| `DATA_DICTIONARY.xlsx` | Every file and column: description, % missing, min / max, most common raw values, and which columns exist in which cycle |
| `download_nhanes.py` | Downloads all 30 files from the CDC (`python download_nhanes.py`) |

The raw data is **not stored in Git**. Get it with `python download_nhanes.py`, or with `dvc pull` once DVC is set up.

## The 15 files per cycle

| File | Topic |
| --- | --- |
| DEMO | Demographics: age, sex, education, income |
| BMX | Body measures: weight, height, BMI, waist |
| BPX (2017) / BPXO (2021) | Blood pressure exam + pulse (manual cuff in 2017, automatic device in 2021) |
| BPQ | Blood pressure / cholesterol questionnaire |
| TCHOL, HDL | Lab: total and HDL cholesterol |
| GHB | Lab: HbA1c (glycohemoglobin) |
| SMQ | Smoking |
| ALQ | Alcohol |
| PAQ | Physical activity |
| SLQ | Sleep |
| DPQ | PHQ-9 depression screener (stress / mental wellbeing) |
| DBQ | Diet behaviour |
| MCQ | Medical conditions (heart disease, heart attack, stroke, ...) |
| DIQ | Diabetes |

## How to load

### Validated and derived data

| Artifact | Rows | Meaning |
| --- | ---: | --- |
| `data/interim/merged_2017-2018.csv` | 5,536 | Original adult, known-label cohort; 31 ordered columns |
| `data/processed/train.csv` / `test.csv` | 4,428 / 1,108 | Original stratified split |
| `data/production/2021-2023_clean.csv` | 7,783 | Later adult cohort; 1,936 positives (24.87%); same column contract |
| `data/production/batches/batch_01.csv` … `batch_06.csv` | 1,298 then 1,297 each | Seed-42 disjoint partitions; simulated weeks, not chronological observations |
| `data/production/batches/batch_shifted.csv` | 1,298 | Synthetic age≥60, BMI+3 and SBP+15 stress data, clipped to API bounds |
| `data/production/train_2021.csv` / `test_2021.csv` | 6,226 / 1,557 | Later stratified split; both test sets excluded from challenger fitting |

The candidate uses 4,428 + 6,226 training rows. Its later test partition must be used for challenger performance/fairness evaluation; full production contains fitting rows. The champion was fitted only on the original training partition, so full later-cycle monitoring is valid for that model. The holdout has been examined during this educational study; further tuning needs training-only validation rather than repeated holdout selection.

`dvc repro` rebuilds all 15 data stages, including later validation, the validated production copy and batches. Challenger splits and artifact are separate `.dvc` outputs created by the opt-in retraining command. `dvc pull` restores both groups; `dvc push` synchronizes their contents. Never commit CSV/XPT files or participant logs to Git.

The synthetic batch retains original labels after changing inputs. Its metrics test software behavior under an artificial shift; they do not estimate what those participants' outcomes would have been after an intervention. `diet_quality` is unavailable in the later cycle and stays excluded from both models. Missing values are preserved for fitted pipeline imputation. See [verification and limitations](docs/phase3-verification.md).

### Read a source file

```python
import pandas as pd
demo = pd.read_sas("data/raw/nhanes/2017-2018/DEMO_J.xpt", format="xport")
```

Files are linked by `SEQN` (one row per person per file). Not every person appears in every file.

## Codebooks (meaning of every column and answer code)

- 2017-2018: `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/<FILE>_J.htm` — [demographics example](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.htm).
- 2021-2023: `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/<FILE>_L.htm` — [demographics example](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.htm).

These current CDC codebook paths were checked on 21 November 2026; the older cycle URLs returned a moved/not-found page during the audit.

Read the codebook before cleaning a column — many numbers are codes, not measurements.

## Source

Centers for Disease Control and Prevention (CDC), National Center for Health Statistics (NCHS). National Health and Nutrition Examination Survey Data. https://wwwn.cdc.gov/nchs/nhanes/
