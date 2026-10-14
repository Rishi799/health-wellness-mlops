# Person 2 Data Dictionary — Body & Clinical Measurements

**Topic Table:** `data/interim/body_<cycle>.csv`  
**Owner:** Rishabh (Person 2: Body, Clinical Measurements & DVC)  
**Survey Cycles Supported:** `2017-2018` (training), `2021-2023` (drift)  
**Key Column:** `seqn` (Unique Respondent Sequence Number)  

---

## 1. Topic Schema Overview

Every row represents exactly one survey participant. All column names strictly follow the snake_case contract approved by the team.

| Column Name | Data Type | Physical Units | Raw Source File(s) | Raw Variable(s) | Description | Allowable Range / Values | Missingness (2017-18) | Missingness (2021-23) |
|---|---|---|---|---|---|---|---|---|
| **`seqn`** | `int64` | ID | DEMO, BMX, BPX, BPQ, TCHOL, HDL, GHB | `SEQN` | Respondent Sequence Number | Positive Integer | 0.0% | 0.0% |
| **`height_cm`** | `float64` | cm | BMX | `BMXHT` | Standing height measured at clinic | $50.0 - 250.0$ | 11.3% | 20.9% |
| **`weight_kg`** | `float64` | kg | BMX | `BMXWT` | Weight measured at clinic | $2.0 - 300.0$ | 5.1% | 18.5% |
| **`bmi`** | `float64` | $\text{kg/m}^2$ | BMX | `BMXBMI` | Body Mass Index | $10.0 - 100.0$ | 11.4% | 21.2% |
| **`waist_cm`** | `float64` | cm | BMX | `BMXWAIST` | Waist circumference | $30.0 - 200.0$ | 15.9% | 23.8% |
| **`sbp`** | `float64` | mmHg | BPX / BPXO | `BPXSY1..4` / `BPXOSY1..3` | Systolic Blood Pressure (mean of valid readings) | $50.0 - 300.0$ | 25.7% | 30.0% |
| **`dbp`** | `float64` | mmHg | BPX / BPXO | `BPXDI1..4` / `BPXODI1..3` | Diastolic Blood Pressure (mean of valid non-zero readings) | $30.0 - 200.0$ | 26.6% | 30.0% |
| **`pulse`** | `float64` | bpm | BPX / BPXO | `BPXPLS` / `BPXOPLS1..3` | Resting pulse (beats per minute) | $30.0 - 220.0$ | 25.4% | 30.0% |
| **`told_high_bp`** | `Int64` | binary flag | BPQ | `BPQ020` | Ever told by doctor had high blood pressure | `1` = Yes, `0` = No | 31.9% | 21.0% |
| **`total_chol`** | `float64` | mg/dL | TCHOL | `LBXTC` | Total serum cholesterol | $50.0 - 600.0$ | 25.4% | 35.9% |
| **`hdl`** | `float64` | mg/dL | HDL | `LBDHDD` | Direct HDL "good" cholesterol | $5.0 - 200.0$ | 25.4% | 35.9% |
| **`hba1c`** | `float64` | % | GHB | `LBXGH` | Glycated hemoglobin (HbA1c) — **EDA ONLY / LEAKAGE RISK** | $3.0 - 25.0$ | 33.1% | 37.5% |

---

## 2. Clinical Classification Reference Bands

For EDA and health awareness score calibration:

### Body Mass Index (`bmi`) — WHO / CDC
- Underweight: $< 18.5$
- Normal weight: $18.5 - 24.9$
- Overweight: $25.0 - 29.9$
- Obesity Class I: $30.0 - 34.9$
- Obesity Class II: $35.0 - 39.9$
- Obesity Class III: $\ge 40.0$

### Blood Pressure (`sbp` / `dbp`) — 2017 ACC/AHA Guidelines
- Normal: $\text{SBP} < 120$ and $\text{DBP} < 80$
- Elevated: $\text{SBP } 120-129$ and $\text{DBP} < 80$
- Stage 1 Hypertension: $\text{SBP } 130-139$ or $\text{DBP } 80-89$
- Stage 2 Hypertension: $\text{SBP} \ge 140$ or $\text{DBP} \ge 90$
- Hypertensive Crisis: $\text{SBP} > 180$ and/or $\text{DBP} > 120$

### Total Cholesterol (`total_chol`) — ATP III
- Desirable: $< 200\text{ mg/dL}$
- Borderline high: $200 - 239\text{ mg/dL}$
- High: $\ge 240\text{ mg/dL}$

### HDL Cholesterol (`hdl`) — ATP III
- Low (major risk factor): $< 40\text{ mg/dL}$ (men), $< 50\text{ mg/dL}$ (women)
- High (protective): $\ge 60\text{ mg/dL}$

### HbA1c (`hba1c`) — ADA
- Normal: $< 5.7\%$
- Prediabetes: $5.7\% - 6.4\%$
- Diabetes: $\ge 6.5\%$ *(Target leakage: do not train on this variable!)*
