# Phase 3 production monitoring and retraining

The implementation follows the P2 and P3 handbooks dated 20 November 2026. Rishabh owns production data, traffic replay, drift evaluation, and the trigger. Rishabh also requested the challenger retraining implementation. Rhea reviews model comparison and promotion, Niranjan provides API logging and dashboards, and Rishi integrates the changes.

## Delivery sequence

| Step | Deliverable | Verification |
| --- | --- | --- |
| 1 | Later-cycle cleaning, validation, and six production partitions | Same ordered 31 columns, disjoint partitions, seed 42, reproducible DVC outputs |
| 2 | Local traffic simulator | Serving features only, HTTP status and latency counts, synthetic validation failures |
| 3 | Evidently and labelled performance reports | 21-feature denominator, separate prediction drift, unchanged quality-gate metric definitions |
| 4 | Retraining trigger | Drift >= 0.30 OR recall < 0.777 OR ROC-AUC < 0.80, including exact boundary tests |
| 5 | Challenger training and comparison | Split first, assert participant separation, evaluate both models on both holdouts, record promotion eligibility |

Each task is developed as a separate reviewable change. The current model artifact, API contract, classification threshold, and quality-gate rules are the release baseline.

## Data and report contracts

Production data is `data/production/2021-2023_clean.csv`. The six files `data/production/batches/batch_01.csv` through `batch_06.csv` partition its rows after shuffling. These partitions simulate weeks and have no real time ordering. `batch_shifted.csv` is a synthetic stress test using older adults and clipped BMI/SBP offsets. The manifest records the transformations and explains that retained labels describe source participants.

Prediction logs follow the handbook’s JSONL contract at `logs/predictions.jsonl`, with an `inputs` object, timestamp, request ID, model version, probability, score, band, and latency. Niranjan’s API work writes this file. The simulator sends traffic, while drift monitoring can consume a CSV immediately and a prediction log when that integration is available.

Evidently compares the original training distribution with later inputs. Dataset drift is calculated from the 21 serving features. Predicted-probability drift is reported separately and does not increase that denominator. A drift report describes a change in distribution. Labelled performance differences can have several explanations, including population and measurement changes, and do not by themselves establish concept drift.

Reports for different batches use separate directories under `reports/drift/`. Match each drift summary and performance report by data fingerprint, model version, and batch name before producing a decision. A batch with a single outcome class has no ROC-AUC estimate. The trigger exits 2 when required metrics are unavailable, and exits 0 for either valid decision. Inspect `retrain_needed` in the JSON rather than interpreting exit 0 as a positive trigger.

The fixed project trigger rules are `share_drifted_features >= 0.30`, `recall < 0.777`, or `roc_auc < 0.80`. Ordinary later-cycle batches may genuinely cross these thresholds. Preserve their observed outcomes. Use identical reference data as a clearly labelled negative control when demonstrating a no-trigger decision.

## Challenger evaluation

Split the later cycle 80/20 with stratification and seed 42 before fitting. Combine only its training partition with the original training set. Keep both test partitions separate. Fit the preprocessing and Logistic Regression using the existing builders, then score both held-out populations.

The comparison reuses the existing quality gate and records its result on each holdout. Eligibility requires hard checks to pass, later-cycle recall at least as high as the champion, later-cycle ROC-AUC within 0.01 of the champion, and original-test ROC-AUC at least 0.80. Equality meets the stated recall and AUC rules. The decision is a review artifact. Deployment follows the team’s reviewed model-release process.

The fixed later holdout has already been evaluated during this educational monitoring study. Repeated tuning against it would weaken its value as a final evaluation. Record that evaluation history and use training-only validation for further model selection.

## Why these tools are included

| Tool | Project use | Output |
| --- | --- | --- |
| Pandera | Validate later-cycle tables before monitoring | `reports/validation_2021-2023.json` |
| DVC | Reproduce cleaned data, partitions, challenger and metadata | `dvc.lock`, later split pointers, `artifacts/retraining.dvc` |
| Evidently 0.7.17 | Compare input and prediction distributions | Interactive HTML and `drift_summary.json` |
| Fairlearn 0.15.0 | Assess recall, precision and selection rate by recorded sex and age | Four held-out assessments in `reports/fairness/` |
| SHAP 0.51.0 | Explain fitted Logistic Regression outputs | Aggregate JSON and importance charts in `reports/explanations/` |
| MLflow / DagsHub | Track candidate parameters, source commit, input fingerprints, model and both holdout metrics | Experiment `cardiometabolic-retraining` |
| GitHub Actions | Check offline behavior, DVC rebuild, tests, quality gate and container startup | CI workflow |

Monitoring packages belong to the development environment. The API uses `requirements-api.txt`, keeping report generation outside request handling.

Evidently uses explicit normalized Wasserstein distance for numerical columns and Jensen-Shannon distance for categorical columns, each with threshold 0.1. This keeps the method stable across batch sizes. Numeric survey codes are canonicalized so `1` and `1.0` do not create false category drift. An entirely missing feature makes distribution drift unavailable; missing fractions are also recorded. The 30% dataset threshold is the team's project rule, rather than Evidently's default.

Fairlearn reports counts, true positives and 95% Wilson recall intervals alongside gaps. These are descriptive, unweighted binomial intervals; they do not account for NHANES's complex survey design. Recorded sex codes 1 and 2 and top-coded age 80 retain the source dataset's meaning. These assessments do not certify fairness or clinical safety. Groups with no positive cases have unavailable recall rather than an invented zero.

SHAP uses `LinearExplainer` with an Independent masker on fitted transformed inputs. Encoded contributions are summed back to the 21 input fields. Contributions are in positive-class log-odds, and an additivity check verifies reconstruction of the model decision function. A seeded sample of training rows supplies the background; the challenger uses the combined training population. These are model attributions under an independence assumption, rather than causal effects. Correlated measurements such as BMI, waist and weight affect attribution interpretation. Published summaries contain aggregate contributions rather than participant-level explanations.

## Reproduce the workflow

Run commands from the repository root in PowerShell with the virtual environment activated. Use a current patched Python 3.11 release. Local validation used an isolated Python 3.11.9 runtime because Python 3.11.0 failed while importing Evidently's typing aliases. The normal environment's package pins were retained. GitHub Actions requests the current Python 3.11 patch release.

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
pip install -r requirements.txt
dvc pull
dvc repro
python -m pytest -q
```

Start the API in one terminal:

```powershell
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

In a second activated terminal, replay real inputs and then append two intentionally invalid requests. `--bad 2` adds requests to those selected by `--limit`.

```powershell
python -m scripts.simulate_traffic --batch data/production/batches/batch_01.csv --limit 20
python -m scripts.simulate_traffic --batch data/production/batches/batch_01.csv --limit 2 --bad 2
```

The local demonstration returned 20/20 HTTP 200 responses, then two HTTP 200 and two HTTP 422 responses, with no transport failures. This was a bounded functional demonstration, not a capacity benchmark. For public endpoints the simulator enforces at most 20 requests at one per second; use localhost for full batches.

Generate a matched report set and trigger with one command:

```powershell
python -m scripts.monitor_batch --current data/production/2021-2023_clean.csv --out reports/drift/full_production
python -m scripts.monitor_batch --current data/production/batches/batch_shifted.csv --out reports/drift/batch_shifted
python -m scripts.monitor_batch --reference data/processed/test.csv --current data/processed/test.csv --out reports/drift/negative_control
```

Open each generated `*_drift.html` in a browser and inspect the JSON beside it. To process each ordinary batch:

```powershell
1..6 | ForEach-Object {
    $batchName = "batch_{0:D2}" -f $_
    python -m scripts.monitor_batch --current "data/production/batches/$batchName.csv" --out "reports/drift/$batchName"
}
```

The component CLIs remain available as `src.monitoring.drift`, `src.monitoring.performance` and `src.monitoring.retrain_trigger`. For unlabelled API JSONL, run drift only. Performance requires observed labels; the combined workflow rejects unlabelled inputs and clears a stale decision before starting.

```powershell
python -m src.monitoring.drift --current logs/predictions.jsonl --out reports/drift/live_logs
```

Train and compare a candidate only after a positive trigger. Set `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` using the local `.env` described in `.env.example`. Never commit credentials. UTF-8 output avoids MLflow's emoji logging failing under the legacy Windows console encoding.

```powershell
python -m src.models.retrain --trigger reports/drift/full_production/trigger.json --log-mlflow
python -m src.models.compare_champion
```

`--log-mlflow` is opt-in; omitting it performs the same local fit and evaluation. `--force` records an explicitly forced experiment. A negative trigger skips training. Model export retains the existing threshold and serving features. Comparison verifies the candidate artifact fingerprint and all five source/split files before evaluating it, rejecting stale combinations.

Generate responsible-model assessments for either holdout:

```powershell
python -m src.monitoring.fairness --current data/production/test_2021.csv --out reports/fairness/champion_2021.json
python -m src.monitoring.fairness --challenger --current data/production/test_2021.csv --out reports/fairness/challenger_2021.json
python -m src.monitoring.explanations --current data/production/test_2021.csv --out reports/explanations/champion
python -m src.monitoring.explanations --challenger --current data/production/test_2021.csv --out reports/explanations/challenger
```

For the original holdout, use `data/processed/test.csv` and distinct output names. Fairlearn and SHAP summaries identify their source data, model fingerprint and MLflow run. After an intentional new fit, refresh the DVC pointers with `dvc add data/production/train_2021.csv data/production/test_2021.csv artifacts/retraining`, upload with `dvc push`, and commit only pointers and aggregate reports.

## Observed results, 20–21 November 2026

The cleaned later cycle has 7,783 adults and 1,936 positive labels (24.87%). All five table validations pass. The six partitions contain 1,298 / 1,297 / 1,297 / 1,297 / 1,297 / 1,297 rows. All seven original-cycle CSV fingerprints remained identical after the complete DVC rebuild.

| Monitoring case | Drifting inputs / 21 | ROC-AUC | Recall | Trigger |
| --- | ---: | ---: | ---: | --- |
| Full later cycle | 8 | 0.8248 | 0.7846 | Data drift |
| Synthetic older/BMI/SBP stress | 13 | 0.7533 | 0.9293 | Data drift and ROC-AUC floor |
| Identical original-test control | 0 | 0.8502 | 0.8270 | No |

All six ordinary batches cross the input-drift rule. Batches 03, 04 and 06 also cross the recall rule. Prediction drift appears in the stress batch and not in the ordinary batches. `reports/drift/batch_results.json` preserves every result. Stress-batch performance uses retained source labels after synthetic transformations, so it is a software stress test rather than an evaluation of counterfactual health outcomes.

The challenger fits 4,428 original plus 6,226 later training rows, holding out 1,108 original and 1,557 later participants. Its source code commit is recorded in training metadata. The successful [DagsHub run](https://dagshub.com/Rishi799/health-wellness-mlops.mlflow/#/experiments/2/runs/893b1b05690d4870ba99b0628355395f) contains the fitted candidate and parameters.

| Holdout | Champion ROC-AUC | Challenger ROC-AUC | Champion recall | Challenger recall |
| --- | ---: | ---: | ---: | ---: |
| 2017–2018 | 0.8502 | 0.8495 | 0.8270 | 0.8062 |
| 2021–2023 | 0.8401 | 0.8457 | 0.8191 | 0.8191 |

The fixed numerical promotion rules pass. The decision still requires review; it does not change the registry alias, API artifact or deployment. Original-holdout recall decreases by about 2.08 percentage points, which should be discussed even though the prescribed original-population floor concerns ROC-AUC.

The most material Fairlearn finding is young-adult recall. On the later holdout, the champion detects 1/14 positive cases aged 20–39 and the challenger detects 3/14. Challenger recall is 21.43%, with a wide descriptive Wilson interval of 7.57%–47.59%. On the original holdout it detects 3/17. The later recorded-sex recall gap is 8.65 percentage points for the challenger versus 7.62 for the champion. Improved aggregate ROC-AUC therefore does not resolve subgroup performance limitations. Keep the educational-use framing and require a substantive model review before release.

The leading challenger SHAP inputs in the seeded later-holdout sample are age, prior high-blood-pressure diagnosis, waist circumference, total cholesterol and PHQ-9 score. Reconstruction error is about `1.1e-15`. Feature importance explains the fitted model's behavior; it does not validate the health meaning of its predictions.

## Delivery and remaining integration

Review the stacked changes in order: production data (PR 28), traffic (PR 29), monitoring (PR 30), retraining (PR 31), then fairness/explanations. Rishi can retarget dependent PRs as their parents merge. The production-data PR's complete CI passed, including the Docker job. Final combined validation is recorded in the review PR.

Final local verification passed all 249 tests, `pip check`, DVC pipeline status and DVC remote status. Both DVC status commands returned empty JSON, confirming an up-to-date pipeline and uploaded tracked artifacts. The SHAP chart was visually inspected. Dependency deprecation warnings remain in upstream libraries; no test failures occurred.

Rishabh's data, traffic, offline monitoring, trigger and candidate implementation are delivered with real evidence. Team Phase 3 completion still requires Niranjan's prediction JSONL and dashboard integration, Rishi's scheduled workflow/alert integration and reviewed merges, and Rhea's model-release review. Retain a human gate between retraining eligibility and champion promotion. The public API continues to serve the Phase 2 champion during this work.

## Reference documentation

- [Evidently DataDriftPreset](https://docs.evidentlyai.com/metrics/preset_data_drift): column selection, explicit types and dataset drift customization.
- [Fairlearn assessment](https://fairlearn.org/main/user_guide/assessment/perform_fairness_assessment.html): assess group harms and interpret disaggregated metrics in context.
- [SHAP LinearExplainer](https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html): linear explanations, masker choice and dependence assumptions.

Project requirements are the local `P2_Rishabh_Phase3_Tasks.pdf` and `P3_Rhea_Phase3_Tasks.pdf` handbooks. The additional fairness/explanation work follows Rishabh's explicit request to integrate these tools where useful.
