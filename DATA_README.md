# Raw data — NHANES (CDC National Health and Nutrition Examination Survey)

Public-domain U.S. health survey data, downloaded unchanged from the CDC. No names or personal identifiers; each person is a number (`SEQN`).

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

```python
import pandas as pd
demo = pd.read_sas("data/raw/nhanes/2017-2018/DEMO_J.xpt", format="xport")
```

Files are linked by `SEQN` (one row per person per file). Not every person appears in every file.

## Codebooks (meaning of every column and answer code)

- 2017-2018: `https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/<FILE>_J.htm` — e.g. https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.htm
- 2021-2023: `https://wwwn.cdc.gov/Nchs/Nhanes/2021-2023/<FILE>_L.htm` — e.g. https://wwwn.cdc.gov/Nchs/Nhanes/2021-2023/DEMO_L.htm

Read the codebook before cleaning a column — many numbers are codes, not measurements.

## Source

Centers for Disease Control and Prevention (CDC), National Center for Health Statistics (NCHS). National Health and Nutrition Examination Survey Data. https://wwwn.cdc.gov/nchs/nhanes/
