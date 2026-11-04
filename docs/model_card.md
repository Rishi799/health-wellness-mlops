# Model card: cardiometabolic-risk-model v2

## Model details

- **Name / version:** cardiometabolic-risk-model / 2 (registered champion).
- **Type:** scikit-learn Logistic Regression pipeline with median imputation and
  standardization for numeric features, most-frequent imputation and one-hot
  encoding for categorical features; 21 input features.
- **Owner:** Health and Wellness MLOps team; Rhea (Person 3) maintains model quality.
- **Card date:** 2026-11-04. **Artifact export:** 2026-11-03.
- **Source run:** `077b778198d448628bf29fbf8c4fd9e6`.
- **Provenance:** `models/model_info.json`; the champion artifact is versioned
  through DVC, not stored in this change.

## Intended use

Awareness and education for adults aged 20 and above. The score describes an
association with self-reported cardiometabolic diagnosis in the study population.
It is **not a diagnosis**, treatment recommendation, or validated estimate of
future disease. It must not replace clinical assessment or determine access to
care. The current API accepts ages 20-80; the dataset caps older ages at 80.

## Training data

NHANES 2017-2018: 5,536 adults with a known label, split into 4,428 training
and 1,108 held-out test rows (stratified 80/20, seed 42). Approximately 26.1%
are positive. No survey weights were used, so results are not survey-weighted
estimates for the US population. Age is capped at 80.

The label is 1 when a participant reports ever being told by a doctor that they
had diabetes (including borderline diabetes), heart failure, coronary heart
disease, angina, heart attack, or stroke; 0 when all six answers are no.
Unknown-target rows are excluded. This is a self-reported diagnosis label,
not a contemporaneous clinical measurement or future event label.

Features exclude `seqn`, `cardiometabolic`, all six target components, `hba1c`,
and `diet_quality`, using the shared `EXCLUDE` set. Numeric missing values
are imputed inside the fitted pipeline; the champion is not refitted by the gate.

## Metrics and quality gate

**Current verification status: pending DVC artifacts.** On 2026-11-04 the clean
checkout could not retrieve the champion or processed data from the configured
DagsHub remote. No new measured results are claimed here. Run
`python -m src.models.quality_gate` after restoring the artifacts, then populate
the following results from `reports/model_quality.json` and inspect
`reports/figures/calibration.png` before merging this work.

Phase 1 reference values recorded in `models/model_info.json` are ROC-AUC
0.8502 and recall 0.827 at threshold 0.5. They are historical reference values,
not a substitute for rerunning the gate on the retrieved champion.

| Overall metric | Current gate value | Agreement |
| --- | --- | --- |
| ROC-AUC | Pending | >= 0.80; hard check |
| Recall | Pending | >= 0.75; hard check |
| Most-frequent dummy ROC-AUC | Pending | Fit baseline on training labels only |
| ROC-AUC improvement over dummy | Pending | >= 0.20; hard check |
| Brier score | Pending | Warning when > 0.20 |

| Subgroup | n | Positives | Recall | Precision | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Sex 1 (male) | Pending | Pending | Pending | Pending | Pending |
| Sex 2 (female) | Pending | Pending | Pending | Pending | Pending |
| Age 20-39 | Pending | Pending | Pending | Pending | Pending |
| Age 40-59 | Pending | Pending | Pending | Pending | Pending |
| Age 60-79 | Pending | Pending | Pending | Pending | Pending |
| Age 80 (capped) | Pending | Pending | Pending | Pending | Pending |

Subgroup recall below 0.60 triggers a warning, not a hard failure. Always
interpret rates with n and positive counts. Recall is unavailable when there
are no positives; ROC-AUC is unavailable with fewer than two target classes.
Empty groups retain n=0 in the JSON report, with unavailable metrics stored as
null. Precision is 0 when a nonempty group has no predicted positives.
These descriptive checks do not establish fairness or statistical certainty.
Brier score and a 10-bin calibration curve assess probability quality; meeting
the Brier threshold does not establish clinical calibration.

The gate also blocks feature leakage, mismatches with `PredictionInput` fields,
and a mismatch between artifact feature ordering and the metadata. All hard
failures are reported together and cause exit code 1. Thresholds may change
only after a team decision.

## Threshold and risk bands

The classification threshold is **0.5**, read from `model_info.json`.
Probability >= 0.5 counts as positive for recall and precision. ROC-AUC uses
the continuous probability and does not depend on this threshold.

The API calculates `risk_score = probability * 100` (rounded to one decimal)
and applies these display bands to that score: Low <30, Moderate >=30 and
<60, High >=60. The band boundaries are distinct from the binary threshold.
These are educational display bands, not clinically validated cutoffs.

## Limitations and risks

- **False negatives:** Phase 1 missed 50 of 289 positive test participants.
  A low score must not reassure someone to dismiss symptoms or delay care.
- **Potential shortcut:** `told_high_bp` may reflect previous healthcare contact
  or existing illness and act as a shortcut for reported diagnosis. Strong
  prediction does not establish a causal relationship.
- **Drift:** NHANES 2021-2023 may differ in population, measurement, missingness,
  and healthcare access. Those data are reserved for later drift evaluation;
  current performance must not be assumed to transfer.
- **Associations, not causes:** neither feature coefficients nor directional
  tests show that changing a lifestyle factor will change an individual's risk.
- Self-report, excluded unknown labels, missing-value imputation, class
  imbalance, no survey weights, and age capping limit generalizability.
  Small subgroup counts make rates uncertain; aggregate success can hide gaps.
- A model of prior diagnosis cannot identify all undiagnosed disease and is not
  validated for children, other populations, or treatment decisions.

## How to report a problem

Open an issue at https://github.com/Rishi799/health-wellness-mlops/issues and tag
the model-quality owner (Rhea / Person 3). Include model name/version, date,
the failing check or endpoint, expected versus observed behavior, and a minimal
synthetic example. Do not post personal health information, credentials, or
identifiable participant data. For a health concern, contact a qualified
healthcare professional; the issue tracker is for software/model problems.
