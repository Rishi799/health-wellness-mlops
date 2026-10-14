# Person 3 - Lifestyle habits: decision log, conversion tables, validation agreement

Source of truth for every rule in `src/data/lifestyle.py` and `src/data/validate.py`.
The `rows` column is filled by the notebook (`notebooks/p3_lifestyle.ipynb`, section 2) once the real raw files are in place.
Check the alcohol wording against the codebook (ALQ_J.htm / ALQ_L.htm) before presenting.

## 1. Decision log

| # | Column | Problem found | What I did | Why |
|---|--------|---------------|------------|-----|
| 1 | all numeric | zeros stored as 5.4e-79 | set abs(x) < 1e-6 to 0 | `== 0` comparisons fail; tiny values look real |
| 2 | SMQ020, SMQ040 | 7 / 9 = refused / don't know | set to NaN | codes, not answers |
| 3 | SMQ040 | blank when SMQ020 = 2 (skip pattern) | SMQ020 = 2 gives `never` | never-smokers are not asked "do you smoke now" |
| 4 | ALQ111 | ALQ111 = 2 (never had a drink) skips the rest | drinks per week = 0 | gate question "no" means 0, not unknown |
| 5 | ALQ121 | frequency category 0-10, not a count | days/year = midpoint of range; times ALQ130 / 52 | averaging codes would be meaningless |
| 6 | ALQ121, ALQ130 | 77/99 and 777/999 = refused / don't know | set to NaN | codes, not values |
| 7 | ALQ121 | 0 = no drinking in the last 12 months (ALQ130 blank) | drinks per week = 0 | skip pattern |
| 8 | PAQ gates | "no" leaves days and minutes blank (2017); count 0 leaves unit and minutes blank (2021) | minutes = 0 | blank means 0 here, not unknown |
| 9 | PAQ days / minutes | 7/9, 77/99, 7777/9999 codes | set to NaN; total is NaN if any part is unknown | do not guess a missing part |
| 10 | exercise_min_week | vigorous vs moderate | vigorous counts double | WHO: 1 vigorous minute is about 2 moderate |
| 11 | exercise_min_week | extreme totals | cap at 3000 min/week (50 h) | keeps the person, removes absurd leverage |
| 12 | PAD790U, PAD810U | units are bytes (`b'W'`), empty when count = 0 | decode; D/W/M/Y multiply by 7, 1, 12/52, 1/52 | same output as 2017 |
| 13 | PAD680 | real values up to 1320 mixed with 7777 / 9999 | codes to NaN | codes would be averaged as minutes |
| 14 | sleep_hours | two columns (workday, weekend), present in both cycles | (5 x SLD012 + 2 x SLD013) / 7; fall back to whichever exists | weights a real week |
| 15 | diet_quality | DBQ700 only in 2017-2018; 7/9 codes | codes to NaN; empty in 2021-2023 | EDA / app tips only, never a model feature |
| 16 | validation | missing-% limit | 40% default; 60% for alcohol_drinks_week; 70% for lab columns; 100% for diet_quality in 2021-2023 | ALQ is adult-only (44.7% / 54.4% missing on the SEQN union); labs are a subsample; DBQ700 is absent in 2021-2023 |

Smoking rule (decision 3, every combination):

| SMQ020 | SMQ040 | smoking_status |
|--------|--------|----------------|
| 2 | blank | never |
| 1 | 1 or 2 | current |
| 1 | 3 | former |
| 7, 9 or blank | any | NaN |
| 1 | 7, 9 or blank | NaN |

## 2. Conversion tables

### 2a. Alcohol: ALQ121 code to days per year

`drinks_week = days_per_year x ALQ130 / 52`.  Never drinkers and code 0 give 0.

| ALQ121 | Wording (confirm in codebook) | Days per year used | Example: 2 drinks per drinking day |
|--------|-------------------------------|--------------------|-------------------------------------|
| 0 | Never in the last year | 0 | 0.00 |
| 1 | Every day | 365 | 14.04 |
| 2 | Nearly every day (5-6 days a week) | 286 | 11.00 |
| 3 | 3 to 4 times a week | 182 | 7.00 |
| 4 | 2 times a week | 104 | 4.00 |
| 5 | Once a week | 52 | 2.00 |
| 6 | 2 to 3 times a month | 30 | 1.15 |
| 7 | Once a month | 12 | 0.46 |
| 8 | 7 to 11 times in the last year | 9 | 0.35 |
| 9 | 3 to 6 times in the last year | 4.5 | 0.17 |
| 10 | 1 to 2 times in the last year | 1.5 | 0.06 |

Check by hand: code 3 with ALQ130 = 2 gives 182 x 2 / 52 = 7.0 drinks per week.

### 2b. Activity

2017-2018: `minutes per domain = days per week x minutes per day`, gated by the yes/no question (no gives 0).
2021-2023: `minutes per domain = times per unit x unit multiplier x minutes each time`.

| Unit | Meaning | Multiplier to times per week | Example |
|------|---------|------------------------------|---------|
| D | per day | 7 | 2 times/day x 7 x 30 min = 420 min/week |
| W | per week | 1 | 3 times/week x 1 x 30 min = 90 min/week |
| M | per month | 12/52 = 0.2308 | 4 times/month x 12/52 x 30 min = 27.7 min/week |
| Y | per year | 1/52 = 0.0192 | 10 times/year x 1/52 x 60 min = 11.5 min/week |

Both cycles: `exercise_min_week = vigorous x 2 + moderate`, capped at 3000.

## 3. Validation agreement (`src/data/validate.py`)

Ranges below are the written agreement with each file owner.  Rows marked CONFIRM need the owner's sign-off.

| Topic | Column | Allowed |
|-------|--------|---------|
| all | seqn | unique, not null, positive integer |
| people (P1) | age | 0-80 in the topic file, 20-80 in the merged file |
| | income_ratio | 0-5 |
| | diabetes, heart_failure, coronary_heart_disease, angina, heart_attack, stroke, cardiometabolic | 0 or 1 |
| | sex, education | present; categories CONFIRM with P1 |
| body (P2) | height_cm 50-230, weight_kg 10-300, bmi 12-80, waist_cm 30-200 | CONFIRM with P2 |
| | sbp 60-260, dbp 20-150, pulse 30-200 | CONFIRM with P2 |
| | told_high_bp | 0 or 1 |
| | total_chol 50-600, hdl 10-200, hba1c 3-20 | CONFIRM with P2 |
| lifestyle (P3) | smoking_status | never, former, current |
| | alcohol_drinks_week | 0-110 |
| | exercise_min_week | 0-3000 |
| | sedentary_min_day | 0-1440 |
| | sleep_hours | 2-14 |
| | diet_quality | 1-5 |
| mental (P4) | phq9_score | 0-27 |
| | phq9_category | present; categories CONFIRM with P4 |

Every numeric column also fails on: a leftover code larger than its maximum (7777, 9999, 999 ...), and float-noise values such as 5.4e-79.
Missing-% limits: default 40%, alcohol_drinks_week 60%, lab columns 70%, diet_quality 100% in 2021-2023.  The target `cardiometabolic` may not be missing in the merged file (confirm with P1 that unknown-target rows are dropped before the merged file is written).

## 4. DVC stage for Person 2

```yaml
  validate:
    cmd: python -m src.data.validate --cycle ${cycle}
    deps:
      - src/data/validate.py
      - data/interim/merged_${cycle}.csv
    outs:
      - reports/validation_${cycle}.json:
          cache: false
```

Replace `${cycle}` with `2017-2018` (or use a `params.yaml` / `foreach` entry).  The command exits with code 1 on any failed check, so `dvc repro` stops here, between merge and split.
The topic files (`people_`, `body_`, `lifestyle_`, `mental_`) are also validated when present; add them to `deps` if you want the stage to rerun when they change.
