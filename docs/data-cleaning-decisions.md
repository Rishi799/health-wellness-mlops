# Person 2 — Clinical & Body Data Cleaning Decision Log

**Author:** Rishabh (Person 2: Body, Clinical Measurements & DVC)  
**Project:** AI-Assisted Health and Wellness Awareness Platform  
**Target Dataset:** `data/interim/body_<cycle>.csv`  
**Cycles Harmonized:** `2017-2018` (Development/Training) & `2021-2023` (Drift Evaluation)  

---

## 1. Executive Summary

This document records all clinical data-engineering decisions, transformation rationales, mathematical aggregation rules, and risk flags implemented in `src/data/body.py`. All transformations prioritize data integrity, biological plausibility, reproducibility, and leakage prevention.

---

## 2. Decision Log Matrix

| Column / Variable | Raw Source Column(s) | Problem Identified | Engineering Action Taken | Clinical & Methodological Justification |
|---|---|---|---|---|
| **`seqn`** | `SEQN` (all files) | Case sensitivity across files; floating-point representation in SAS | Cast to standard lowercase integer `int64`. Assert uniqueness (`assert body['seqn'].is_unique`). | `seqn` is the universal foreign key across all NHANES tables and topic datasets. |
| **`sbp`** | `BPXSY1`..`BPXSY4` (2017) / `BPXOSY1`..`BPXOSY3` (2021) | Up to 4 repeated measurements per person; incomplete readings; reading 1 anxiety spike | Take arithmetic mean of all valid non-zero readings. Round to 1 decimal place. | Averaging multiple valid seated readings mitigates intra-individual measurement variance. NHANES already mandates a 5-minute pre-test rest period, so all valid readings are considered. |
| **`dbp`** | `BPXDI1`..`BPXDI4` (2017) / `BPXODI1`..`BPXODI3` (2021) | Diastolic values recorded as `0` mmHg (stored as `5.4e-79` in SAS) | Any DBP `<= 0` is treated as invalid and converted to `NaN` prior to averaging. SBP must be strictly greater than DBP. | Diastolic blood pressure of 0 mmHg in an ambulatory survey participant is an auscultation/cuff artifact (inability to detect Korotkoff phase 5), not a physiological true zero. |
| **`pulse`** | `BPXPLS` (2017) / `BPXOPLS1`..`BPXOPLS3` (2021) | In 2017, single 60-second pulse; in 2021, 3 readings from automated device | In 2017, take `BPXPLS`. In 2021, compute mean of `BPXOPLS1`..`3`. Filter non-positive values. | Harmonizes manual radial pulse check (2017) with automated oscillometric pulse readings (2021) under a single contract schema. |
| **`told_high_bp`** | `BPQ020` | Categorical survey encoding: `1` = Yes, `2` = No, `7` = Refused, `9` = Don't Know | Map `1 -> 1`, `2 -> 0`, and `7, 9 -> NaN`. Store as pandas nullable integer (`Int64`). | Standardizes boolean contract (1/0). Refusals and unknown responses cannot be assumed to be "No". |
| **`height_cm`**, **`weight_kg`**, **`waist_cm`** | `BMXHT`, `BMXWT`, `BMXWAIST` | Missing values where clinic exam was skipped; SAS near-zeros | Select values, enforce `> 0`, preserve missingness as `NaN`. Round to 1 decimal place. | True anthropometric measurements must be strictly positive. Missing values are preserved for downstream pipeline imputation. |
| **`bmi`** | `BMXBMI` | Potential discrepancy between reported BMI and calculated `weight / (height/100)^2` | Analyzed difference: max difference is `< 0.05` kg/m² (due to CDC rounding). Retained CDC reported `BMXBMI`. | Preserves authoritative CDC examination standard while confirming internal consistency. |
| **`total_chol`**, **`hdl`** | `LBXTC`, `LBDHDD` | Lab tests performed only on a subsample of participants who provided blood samples | Enforce `> 0`, preserve genuine missingness as `NaN`. Do not impute. | Missingness in lab tests is not missing completely at random (subsample protocol); imputation must happen strictly within train splits. |
| **`hba1c`** | `LBXGH` | Direct marker of glycated hemoglobin; clinical diagnostic threshold for diabetes | Cleaned, validated, and retained in topic table for EDA, **with explicit target leakage flag**. | Clinical diagnostic criterion: HbA1c ≥ 6.5% defines diabetes. Diabetes is a component of the prediction target `cardiometabolic`. Retained for EDA, but **must be excluded from model features**. |

---

## 3. Detailed Clinical Rationale & Protocols

### 3.1 Blood Pressure Harmonization & Device Methodology
- **2017–2018 Protocol:** Blood pressure was measured using a manual mercury sphygmomanometer by a trained physician. Up to four readings were attempted. Diastolic pressure was recorded at Korotkoff phase 5 (disappearance of sound). When the sound was heard down to 0, or the measurement could not be determined, 0 was entered. We identified 134 such readings in reading 1, 124 in reading 2, 127 in reading 3, and 9 in reading 4. These were removed from the participant's average.
- **2021–2023 Protocol:** NHANES transitioned to the Omron HEM-907XL automated oscillometric device. Three readings were taken automatically at 1-minute intervals. No manual Korotkoff sound interpretation was involved.
- **Harmonized Aggregation Rule:**
  $$\text{sbp} = \frac{1}{k} \sum_{i \in \text{valid}} \text{SBP}_i, \quad \text{dbp} = \frac{1}{k} \sum_{i \in \text{valid}} \text{DBP}_i$$
  where $k$ is the count of valid readings ($k \ge 1$). If no valid readings exist, both are set to `NaN`.

### 3.2 Target Leakage Analysis: `hba1c`
- **Clinical definition:** Glycated hemoglobin ($\text{HbA}_{1\text{c}}$) reflects average blood glucose levels over the preceding 2–3 months.
- **Diagnostic criteria (American Diabetes Association):**
  - Normal: $< 5.7\%$
  - Prediabetes: $5.7\% - 6.4\%$
  - Diabetes: $\ge 6.5\%$
- **The Leakage Threat:** Person 1 defines the prediction target `cardiometabolic = 1` if a participant has ever been told they have diabetes (`DIQ010`) or specific cardiovascular conditions. Since $\text{HbA}_{1\text{c}} \ge 6.5\%$ is the physiological definition of diabetes, any model trained with `hba1c` as a predictor will learn a trivial shortcut rather than lifestyle/awareness risk factors.
- **Team Directive:** `hba1c` is retained in `body_<cycle>.csv` for medical validation and exploratory analysis, but **Person 4 must drop `hba1c` from the training feature matrix**.

### 3.3 Clinical Risk Flag: `told_high_bp`
- `told_high_bp` originates from questionnaire `BPQ020`.
- While high blood pressure is a major risk factor for cardiometabolic disease, it is not part of Person 1's target definition (which includes heart failure, coronary heart disease, angina, heart attack, stroke, and diabetes).
- Recommendation to Person 4: Evaluate model performance with and without `told_high_bp` to ensure the model does not rely solely on prior clinical diagnoses.
