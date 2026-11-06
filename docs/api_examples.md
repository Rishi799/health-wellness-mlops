# Cardiometabolic Risk Assessment API — Documentation & Examples

Production inference service serving the champion Logistic Regression model trained on NHANES data.

- **Base URL:** `http://127.0.0.1:8000` (default local port)
- **Model Name:** `cardiometabolic-risk-model`
- **Champion Version:** `2`
- **Framework:** FastAPI / Pydantic v2 / scikit-learn / skops

---

## 1. Endpoints Overview

| Method | Endpoint | Description | Success Code | Error Codes |
|---|---|---|---|---|
| `GET` | `/health` | Check service health and verify model artifact readiness | `200 OK` | `503 Service Unavailable` |
| `GET` | `/model-info` | Retrieve champion model provenance, metadata, metrics, and feature contract | `200 OK` | `503 Service Unavailable` |
| `POST` | `/predict` | Predict probability, risk score, and risk band for a patient | `200 OK` | `422 Unprocessable Entity`, `503 Service Unavailable` |

---

## 2. Feature Contract Specification

The inference pipeline accepts 21 features in the exact ordering expected by the champion pipeline.

| # | Feature Name | Data Type | Required? | Allowed Range / Values | Description |
|:---:|---|:---:|:---:|---|---|
| 1 | `age` | `int` | **Yes** | `20` – `80` | Age in years (adult cohort) |
| 2 | `sex` | `int` | **Yes** | `1` (Male), `2` (Female) | Biological sex |
| 3 | `education` | numeric category | No | Exactly `1`, `2`, `3`, `4`, `5` (equivalent floats accepted) | 1: <9th grade, 2: 9–11th, 3: High school/GED, 4: Some college/AA, 5: College grad+ |
| 4 | `income_ratio` | `float` | No | `0.0` – `5.0` | Family income-to-poverty ratio |
| 5 | `height_cm` | `float` | No | `50.0` – `230.0` | Standing height in cm |
| 6 | `weight_kg` | `float` | No | `2.0` – `300.0` | Measured weight in kg |
| 7 | `bmi` | `float` | No | `12.0` – `80.0` | Body Mass Index ($\text{kg/m}^2$) |
| 8 | `waist_cm` | `float` | No | `30.0` – `200.0` | Waist circumference in cm |
| 9 | `sbp` | `float` | No | `60.0` – `260.0` | Systolic blood pressure (mmHg) |
| 10 | `dbp` | `float` | No | `20.0` – `150.0` | Diastolic blood pressure (mmHg) |
| 11 | `pulse` | `float` | No | `30.0` – `200.0` | Resting pulse (beats per minute) |
| 12 | `told_high_bp` | `float` | No | `0.0` (No), `1.0` (Yes) | Ever told by doctor had high blood pressure |
| 13 | `total_chol` | `float` | No | `50.0` – `600.0` | Total serum cholesterol (mg/dL) |
| 14 | `hdl` | `float` | No | `10.0` – `200.0` | Direct HDL cholesterol (mg/dL) |
| 15 | `smoking_status` | `string` | No | `"never"`, `"former"`, `"current"` | Cigarette smoking history |
| 16 | `alcohol_drinks_week`| `float` | No | `0.0` – `110.0` | Alcoholic beverages consumed per week |
| 17 | `exercise_min_week` | `float` | No | `0.0` – `3000.0` | Moderate/vigorous physical activity (min/week) |
| 18 | `sedentary_min_day` | `float` | No | `0.0` – `1440.0` | Sedentary minutes per day |
| 19 | `sleep_hours` | `float` | No | `2.0` – `14.0` | Typical daily sleep duration |
| 20 | `phq9_score` | `float` | No | `0.0` – `27.0` | PHQ-9 depression screening score |
| 21 | `phq9_category` | `string` | No | `"minimal"`, `"mild"`, `"moderate"`, `"moderately severe"`, `"severe"` | PHQ-9 severity category |

> [!NOTE]
> All optional fields default to `null` if omitted. Missing values are imputed using the champion pipeline's preprocessing transformers (median for numeric features, most frequent for categorical features). Extra unknown fields are strictly forbidden (`extra="forbid"`).

---

## 3. Risk Stratification Bands

| Risk Band | Risk Score Range | Approximate Model Output Range | Project Display Meaning |
|---|---|---|---|
| **Low** | `< 30.0` | `< 0.3000` | Low awareness band |
| **Moderate** | `30.0` – `59.9` | `0.3000` – `0.5999` | Moderate awareness band |
| **High** | `≥ 60.0` | `≥ 0.6000` | High awareness band |

Bands are applied after rounding the score to one decimal. These are project-defined display bands, not clinically validated thresholds. The model is uncalibrated and predicts association with prior self-reported diagnosis, not future disease. See the model card, including the 2/17 recall finding for ages 20-39. Boolean values are rejected for all inputs; fractional education and binary BP-history values return 422. Missing optional values remain supported.

`GET /model-info` also returns `service_git_commit` for the application build. Its `git_commit` field remains the model training commit. Public 500/503 errors omit internal exception details; use private server logs for diagnosis.

---

## 4. Endpoint Examples & cURL Requests

### Example 1: System Health Check (`GET /health`)

```bash
curl -X GET "http://127.0.0.1:8000/health" \
     -H "Accept: application/json"
```

**Response (`200 OK`):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "2",
  "error": null
}
```

If the model artifact failed to load, `/health` returns `503 Service Unavailable`:
```json
{
  "status": "unhealthy",
  "model_loaded": false,
  "model_version": null,
  "error": "Model is unavailable"
}
```

---

### Example 2: Model Information (`GET /model-info`)

```bash
curl -X GET "http://127.0.0.1:8000/model-info" \
     -H "Accept: application/json"
```

**Response (`200 OK`):**
```json
{
  "model_name": "cardiometabolic-risk-model",
  "model_version": "2",
  "source_run_id": "077b778198d448628bf29fbf8c4fd9e6",
  "git_commit": "003e6893762a84d72f6b66d4874a705ab3584560",
  "features": [
    "age",
    "sex",
    "education",
    "income_ratio",
    "height_cm",
    "weight_kg",
    "bmi",
    "waist_cm",
    "sbp",
    "dbp",
    "pulse",
    "told_high_bp",
    "total_chol",
    "hdl",
    "smoking_status",
    "alcohol_drinks_week",
    "exercise_min_week",
    "sedentary_min_day",
    "sleep_hours",
    "phq9_score",
    "phq9_category"
  ],
  "decision_threshold": 0.5,
  "metrics": {
    "test_roc_auc": 0.8502,
    "test_recall": 0.827
  },
  "test_roc_auc": 0.8502,
  "test_recall": 0.827,
  "export_date": "2026-11-03T14:05:15.820079+00:00"
}
```

---

### Example 3: Low-Risk Patient (`POST /predict`)

Young adult (age 25, female), healthy BMI (21.3), optimal blood pressure (110/70 mmHg), desirable cholesterol, non-smoker, physically active.

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 25,
       "sex": 2,
       "education": 4.0,
       "income_ratio": 4.5,
       "height_cm": 165.0,
       "weight_kg": 58.0,
       "bmi": 21.3,
       "waist_cm": 72.0,
       "sbp": 110.0,
       "dbp": 70.0,
       "pulse": 68.0,
       "told_high_bp": 0.0,
       "total_chol": 170.0,
       "hdl": 65.0,
       "smoking_status": "never",
       "alcohol_drinks_week": 0.0,
       "exercise_min_week": 240.0,
       "sedentary_min_day": 240.0,
       "sleep_hours": 8.0,
       "phq9_score": 1.0,
       "phq9_category": "minimal"
     }'
```

**Response (`200 OK`):**
```json
{
  "probability": 0.0354,
  "risk_score": 3.5,
  "risk_band": "Low",
  "model_version": "2",
  "disclaimer": "This tool is for informational and educational purposes only and does not constitute medical advice, diagnosis, or treatment. Consult a licensed healthcare provider for clinical evaluation."
}
```

---

### Example 4: High-Risk Patient (`POST /predict`)

Older adult (age 68, male), stage 2 hypertension (155/95 mmHg, told high BP), obesity class I (BMI 34.3, waist 112 cm), elevated cholesterol (260 mg/dL), low HDL (35 mg/dL), current smoker, sedentary.

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 68,
       "sex": 1,
       "education": 2.0,
       "income_ratio": 1.2,
       "height_cm": 175.0,
       "weight_kg": 105.0,
       "bmi": 34.3,
       "waist_cm": 112.0,
       "sbp": 155.0,
       "dbp": 95.0,
       "pulse": 85.0,
       "told_high_bp": 1.0,
       "total_chol": 260.0,
       "hdl": 35.0,
       "smoking_status": "current",
       "alcohol_drinks_week": 14.0,
       "exercise_min_week": 0.0,
       "sedentary_min_day": 600.0,
       "sleep_hours": 5.0,
       "phq9_score": 16.0,
       "phq9_category": "moderately severe"
     }'
```

**Response (`200 OK`):**
```json
{
  "probability": 0.8755,
  "risk_score": 87.6,
  "risk_band": "High",
  "model_version": "2",
  "disclaimer": "This tool is for informational and educational purposes only and does not constitute medical advice, diagnosis, or treatment. Consult a licensed healthcare provider for clinical evaluation."
}
```

---

### Example 5: Moderate-Risk Patient (`POST /predict`)

Middle-aged adult (age 52, male), prehypertension (128/82 mmHg), overweight (BMI 27.7), former smoker, borderline cholesterol (210 mg/dL).

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 52,
       "sex": 1,
       "education": 3.0,
       "income_ratio": 2.5,
       "height_cm": 172.0,
       "weight_kg": 82.0,
       "bmi": 27.7,
       "waist_cm": 92.0,
       "sbp": 128.0,
       "dbp": 82.0,
       "pulse": 74.0,
       "told_high_bp": 0.0,
       "total_chol": 210.0,
       "hdl": 45.0,
       "smoking_status": "former",
       "alcohol_drinks_week": 4.0,
       "exercise_min_week": 90.0,
       "sedentary_min_day": 360.0,
       "sleep_hours": 7.0,
       "phq9_score": 6.0,
       "phq9_category": "mild"
     }'
```

**Response (`200 OK`):**
```json
{
  "probability": 0.3838,
  "risk_score": 38.4,
  "risk_band": "Moderate",
  "model_version": "2",
  "disclaimer": "This tool is for informational and educational purposes only and does not constitute medical advice, diagnosis, or treatment. Consult a licensed healthcare provider for clinical evaluation."
}
```

---

### Example 6: Partial Input with Missing Values (`POST /predict`)

Only required demographics are provided. Missing clinical/lifestyle values are automatically imputed by the pipeline.

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 45,
       "sex": 1
     }'
```

**Response (`200 OK`):**
```json
{
  "probability": 0.2847,
  "risk_score": 28.5,
  "risk_band": "Low",
  "model_version": "2",
  "disclaimer": "This tool is for informational and educational purposes only and does not constitute medical advice, diagnosis, or treatment. Consult a licensed healthcare provider for clinical evaluation."
}
```

---

### Example 7: Validation Error Handling (`HTTP 422`)

Input failing contract validation rules (e.g., `age: 19` outside allowable range `20–80`, or invalid `smoking_status: "vaping"`):

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 19,
       "sex": 1,
       "smoking_status": "vaping"
     }'
```

**Response (`422 Unprocessable Entity`):**
```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["body", "age"],
      "msg": "Input should be greater than or equal to 20",
      "input": 19,
      "ctx": {
        "ge": 20
      }
    },
    {
      "type": "literal_error",
      "loc": ["body", "smoking_status"],
      "msg": "Input should be 'never', 'former' or 'current'",
      "input": "vaping",
      "ctx": {
        "expected": "'never', 'former' or 'current'"
      }
    }
  ]
}
```

