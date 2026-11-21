# Model card: cardiometabolic-risk-model v2

## Model details

- **Name / version:** cardiometabolic-risk-model / 2 (registered champion).
- **Type:** scikit-learn Logistic Regression pipeline with median imputation and
  standardization for numeric features, most-frequent imputation and one-hot
  encoding for categorical features; 21 input features.
- **Owner:** Health and Wellness MLOps team; Rhea (Person 3) maintains model quality.
- **Card updated:** 2026-11-21 (Phase 3 evaluation addendum). **Artifact export:** 2026-11-03.
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

**Gate result: PASS with one subgroup warning**, evaluated on 2026-11-04.
Results below come from `reports/model_quality.json`, using the registered
champion and all 1,108 held-out test rows (289 positives), at threshold 0.5.
The downloaded artifact matches its DVC MD5
`2b3f51261f58027084ae18a3144576bc`; train/test content matches the DVC lock
hashes after normalizing Windows line endings. The champion was not refitted.
Reproduce with `python -m src.models.quality_gate`; see the 10-bin calibration
curve in `reports/figures/calibration.png`. Values are rounded to four decimals
here; the JSON report retains full precision.

| Overall metric | Current gate value | Agreement |
| --- | --- | --- |
| ROC-AUC | 0.8502 | >= 0.80; hard check |
| Recall | 0.8270 | >= 0.75; hard check |
| Precision | 0.5494 | Descriptive; no gate threshold |
| Most-frequent dummy ROC-AUC | 0.5000 | Fit baseline on training labels only |
| ROC-AUC improvement over dummy | 0.3502 | >= 0.20; hard check |
| Brier score | 0.1617 | Warning when > 0.20 |

| Subgroup | n | Positives | Recall | Precision | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Sex 1 (male) | 542 | 142 | 0.8873 | 0.5431 | 0.8760 |
| Sex 2 (female) | 566 | 147 | 0.7687 | 0.5567 | 0.8293 |
| Age 20-39 | 358 | 17 | 0.1176 | 0.6667 | 0.7254 |
| Age 40-59 | 319 | 55 | 0.7273 | 0.5063 | 0.7956 |
| Age 60-79 | 345 | 174 | 0.8851 | 0.5725 | 0.7266 |
| Age 80 (capped) | 86 | 43 | 1.0000 | 0.5119 | 0.6858 |

**Subgroup finding:** ages 20-39 trigger the recall warning: only 2 of 17
positives were detected (15 missed) among 358 participants. This low recall
requires team review despite the overall gate passing. The other five groups
meet the 0.60 recall warning threshold. Female recall (0.7687) is lower than
male recall (0.8873); the descriptive comparison does not establish a cause.

Subgroup recall below 0.60 triggers a warning, not a hard failure. Always
interpret rates with n and positive counts. Recall is unavailable when there
are no positives; ROC-AUC is unavailable with fewer than two target classes.
Empty groups retain n=0 in the JSON report, with unavailable metrics stored as
null. Precision is 0 when a nonempty group has no predicted positives.
These descriptive checks do not establish fairness or statistical certainty.
Brier score and a 10-bin calibration curve assess probability quality; meeting
the Brier threshold does not establish clinical calibration. In this run, all
ten occupied bins have mean predicted probability above the observed positive
fraction, indicating overestimation on this test set despite the passing Brier
check.

The gate also blocks feature leakage, mismatches with `PredictionInput` fields,
and a mismatch between artifact feature ordering and the metadata. All hard
failures are reported together and cause exit code 1. Thresholds may change
only after a team decision.

## Phase 3 evaluation and challenger review — 21 November 2026

The served champion is still version 2. Its later-cycle evaluation now uses 7,783 adult records with 1,936 positive labels; all were unseen during its fitting. The unchanged champion achieves ROC-AUC 0.8248, recall 0.7846, precision 0.4848 and Brier 0.1759 on that full cohort. Eight of 21 input distributions drift relative to the original training data. All six ordinary later-cycle batches cross the fixed drift trigger; an identical-data negative control produces no trigger.

For challenger comparison, the later cohort is separately split 6,226/1,557 with stratification and seed 42. Only the later training portion joins the 4,428 original training rows. Both holdouts remain disjoint from fitting and preprocessing. The full later cohort contains candidate fitting rows and is therefore not used as a candidate test set.

| Model / holdout | ROC-AUC | Recall | Precision | Brier |
| --- | ---: | ---: | ---: | ---: |
| Champion v2 / original | 0.8502 | 0.8270 | 0.5494 | 0.1617 |
| Challenger v3 / original | 0.8495 | 0.8062 | 0.5521 | 0.1608 |
| Champion v2 / later holdout | 0.8401 | 0.8191 | 0.4930 | 0.1723 |
| Challenger v3 / later holdout | 0.8457 | 0.8191 | 0.5056 | 0.1658 |

Version 3 is registered with alias `challenger`, from finished run `893b1b05690d4870ba99b0628355395f`. It passes the prescribed numerical eligibility conditions, but is not approved or deployed as champion. Version 2's `champion` alias was verified remotely during the audit. Original-holdout recall decreases by about 2.08 percentage points and must be considered in review.

Fairlearn's later-holdout age 20–39 counts are 1/14 positives detected by the champion and 3/14 by the candidate. Candidate recall is 0.2143 with a descriptive Wilson 95% interval of 0.0757–0.4759. The candidate detects 3/17 original-holdout young-adult positives. The later recorded-sex recall gap is 0.0865 for the candidate versus 0.0762 for the champion. These findings do not establish fairness or resolve the young-adult limitation. The intervals ignore the NHANES complex survey design and reflect small subgroup counts.

SHAP `LinearExplainer` uses training-only backgrounds, explains positive-class log-odds, groups encoded contributions into the 21 serving features and verifies additivity. Its attribution assumptions and correlated measurements prevent a causal interpretation. No individual explanation records are published in the aggregate summary.

Evidence: [full comparison](../reports/retraining/comparison.json), [decision](../reports/retraining/promotion_decision.json), [Fairlearn](../reports/fairness/), [SHAP](../reports/explanations/) and the [requirement/evaluation audit](phase3-verification.md). The later holdout has already been inspected in this educational study; further tuning must use training-only validation. Live JSONL monitoring, dashboards and scheduled integration remain pending; current offline reports do not prove that the full live feedback loop operates.

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
  The current gate reproduces that overall recall, and misses 15 of 17
  positives in the 20-39 age group.
  A low score must not reassure someone to dismiss symptoms or delay care.
- **Potential shortcut:** `told_high_bp` may reflect previous healthcare contact
  or existing illness and act as a shortcut for reported diagnosis. Strong
  prediction does not establish a causal relationship.
- **Drift:** the 2021-2023 evaluation above shows input and performance differences.
  Population, measurement, missingness and healthcare-access changes may contribute;
  the observed drift does not establish a cause or guarantee transfer elsewhere.
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
