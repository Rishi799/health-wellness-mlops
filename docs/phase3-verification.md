# Rishabh's Phase 3 completion audit

Verified on **21 November 2026 (IST)** against the four P2 handbook tasks and the additionally authorized retraining/Fairlearn/SHAP work. Machine-readable evidence uses UTC timestamps. The implementation is ready for teammate review; the live prediction-logging handoff is still pending. This is a point-in-time audit, not a claim that the whole team's Phase 3 is finished.

## Requirement-by-requirement status

| Requirement | Result and evidence | Status |
| --- | --- | --- |
| Later-cycle cleaning, ordered 31-column contract and validation | 7,783 adults, 1,936 positives; five table validations pass | Implemented and verified |
| Original data preserved | Seven original CSV fingerprints match; source and champion files preserved | Verified |
| Six batches and synthetic shift | Every production row appears exactly once in six ordinary batches; separate synthetic 1,298-row stress batch | Implemented and verified |
| DVC reproducibility/upload | 15 data stages; `dvc status --json` and `dvc status --cloud --json` both `{}`; candidate/splits uploaded | Verified |
| Payload conversion and complete local replay | All 9,081 ordinary/stress payloads validate; batch_01 replay gives 1,298/1,298 HTTP 200; separate invalid run gives 5/5 HTTP 422 | Implemented and verified |
| Successful predictions appear in shared JSONL | Current API has no prediction-log writer or log volume | Pending Niranjan's API integration |
| Evidently CSV/log reader, feature/prediction drift | Nine internally consistent report sets; full production 8/21 drifting inputs; separate prediction drift; HTML opened in browser | Implemented and verified for available data |
| Normal-batch no-drift expectation | All six ordinary later-cycle batches actually drift | Expected demonstration story not met; observed results preserved |
| Labelled degradation experiment | Overall and subgroup tables below; metrics recomputed independently | Complete |
| Three-rule trigger and both outcomes | Synthetic stress triggers; identical-data control does not; boundary/malformed-input tests pass; screenshots saved | Implemented and verified |
| Challenger fit and comparison | 10,654 training rows; no overlap with either holdout; same preprocessing/LR builders and threshold | Implemented and verified |
| Registry handoff | Successful run, version 3 `challenger`; version 2 remains `champion` | Verified remotely |
| Fairlearn and SHAP | Four holdout assessments; model/data fingerprints; 21 grouped SHAP fields and additivity check | Implemented and verified |
| Teammate review and merge | PRs 28→32 remain open; candidate release requires review | Pending team action |

The simulator's code and local HTTP behavior are complete. Its strict handbook “done when” also asks for generated prediction-log lines, so that integration criterion must stay open. Do not claim that live log-to-drift monitoring or dashboards have been demonstrated yet.

## Evidence checked

- [Independent numerical verification](../reports/phase3/numerical_verification.json): participant separation, split reproduction, data/model fingerprints, all payloads, nine report sets, quality decision, Fairlearn/SHAP provenance and original data hashes. Recomputed overall metrics match the recorded values exactly in this audit (`max_absolute_error = 0.0`).
- [Complete local replay](../reports/phase3/local_replay.json): full first-batch HTTP results and five invalid requests. Mean local valid-request latency was 14.844 ms at a configured sequential rate of 25 requests/s. This is a functional replay, not a concurrency, production latency or capacity benchmark.
- [Remote verification](../reports/phase3/remote_verification.json): open PR chain, registry aliases and finished MLflow run. [Public API verification](../reports/phase3/public_api.json): `/health`, `/model-info`, `/docs` and one synthetic prediction all returned 200, serving model 2 and application commit `54727be`.
- [Batch drift results](../reports/drift/batch_results.json), [comparison](../reports/retraining/comparison.json), [promotion decision](../reports/retraining/promotion_decision.json), [Fairlearn reports](../reports/fairness/) and [SHAP summaries](../reports/explanations/).
- [No-trigger screenshot](../reports/phase3/screenshots/no-retrain.jpg), [positive-trigger screenshot](../reports/phase3/screenshots/retrain-requested.jpg) and [Evidently screenshot](../reports/phase3/screenshots/production-drift.jpg). The two decision views are generated from the corresponding committed JSON; they are not substitute experiment runs. Open the full Evidently HTML locally for the interactive distributions.

All **256 local tests pass** after the audit fixes. The prior implementation commit passed all three [combined CI jobs](https://github.com/Rishi799/health-wellness-mlops/actions/runs/44803814445), including 253 tests. The updated final-head CI is linked from [PR 32](https://github.com/Rishi799/health-wellness-mlops/pull/32). `pip check` reports no broken requirements. Upstream dependency deprecation warnings remain.

## Degradation experiment: the unchanged champion

This comparison evaluates the same model at threshold 0.5 on the original holdout and the full later cohort. The later cohort was unseen during champion fitting. It is distinct from the 1,557-row challenger holdout used for candidate comparison.

| Metric | 2017–2018 holdout | Full 2021–2023 cohort | Later minus original |
| --- | ---: | ---: | ---: |
| n | 1,108 | 7,783 | — |
| Positive labels | 289 | 1,936 | — |
| ROC-AUC | 0.850163 | 0.824781 | −0.025382 |
| Recall | 0.826990 | 0.784607 | −0.042382 |
| Precision | 0.549425 | 0.484839 | −0.064586 |
| Brier score (lower is better) | 0.161677 | 0.175919 | +0.014242 |

| Group | Original n / positives | Original recall | Later n / positives | Later recall |
| --- | ---: | ---: | ---: | ---: |
| Recorded sex 1 | 542 / 142 | 0.8873 | 3,473 / 957 | 0.8339 |
| Recorded sex 2 | 566 / 147 | 0.7687 | 4,310 / 979 | 0.7365 |
| Age 20–39 | 358 / 17 | 0.1176 | 2,082 / 104 | 0.1058 |
| Age 40–59 | 319 / 55 | 0.7273 | 2,216 / 464 | 0.5280 |
| Age 60–79 | 345 / 174 | 0.8851 | 2,966 / 1,113 | 0.9066 |
| Age 80, top-coded | 86 / 43 | 1.0000 | 519 / 255 | 0.9961 |

Ranking, recall and precision decrease, while Brier error increases on the later cohort. Aggregate recall falls by about 4.24 percentage points, remaining above the project's 0.777 trigger floor; ROC-AUC remains above 0.80. The full-cohort trigger fires because 8/21 input distributions cross the drift thresholds. Ages 20–39 remain poorly detected, and recall for ages 40–59 falls to 0.5280. These descriptive differences motivate review, but population composition, measurement and missingness changes mean they do not independently prove concept drift or its cause.

## Ordinary drift is a result to explain

The full-cohort drifting fields are age, income ratio, SBP, DBP, weekly alcohol, weekly exercise, sedentary minutes and PHQ-9 score. The largest normalized distance is exercise minutes (0.3749). Missingness also changes: for example, BMI missingness rises from 6.87% in the reference to 23.53% in the later cohort; SBP from 10.00% to 24.89%. Missingness is reported separately and is not an additional trigger under the agreed rules.

Survey comparability is a plausible contributor, rather than a proven explanation. This pipeline uses [2017 recreational activity days/minutes](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/PAQ_J.htm) versus [later frequency/unit/minutes questions](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/PAQ_L.htm), and [2017 BPX](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPX_J.htm) versus [later oscillometric measurements](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BPXO_L.htm). CDC also describes an [updated later-cycle sample design and procedures](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.htm). The implementation retains the agreed cleaning paths and original hashes. Review these factors with Rishi/Rhea before changing definitions or alert thresholds.

The identical-reference control proves the zero-change case. It must be presented as a control, not as evidence that ordinary later-cycle batches are stable. Stress-batch labels are retained after artificial input changes, so its ROC-AUC is stress-test evidence only.

## Candidate result and release decision

The 6,226/1,557 later split is reproducible and disjoint; its training portion joins the 4,428 original training rows. Both holdouts are excluded from fitting, preprocessing and SHAP background construction. Saved candidate provenance matches all five input files and the skops artifact. All model comparisons use the same holdouts.

| Model and holdout | ROC-AUC | Recall | Precision | Brier |
| --- | ---: | ---: | ---: | ---: |
| Champion, original | 0.8502 | 0.8270 | 0.5494 | 0.1617 |
| Challenger, original | 0.8495 | 0.8062 | 0.5521 | 0.1608 |
| Champion, later holdout | 0.8401 | 0.8191 | 0.4930 | 0.1723 |
| Challenger, later holdout | 0.8457 | 0.8191 | 0.5056 | 0.1658 |

The numerical eligibility result is correct under the fixed rules. It is not deployment approval. Young-adult challenger recall is 3/14 on the later holdout (21.43%, descriptive Wilson interval 7.57%–47.59%) and 3/17 on the original holdout. Original overall recall decreases by about 2.08 percentage points, and the later recorded-sex recall gap increases slightly. The project remains educational; aggregate improvement does not resolve these limitations. Model version 3 is stored for review, while version 2 remains champion.

## Operational limits and fixes

The audit fixed stale component outputs: a failed standalone drift or performance run now removes its prior output; the combined monitor removes the entire current report set and decision before a rerun. Regression tests cover these failure paths. Data/model fingerprint checks continue to reject mixed report pairs and stale candidate artifacts.

Valid API calls may omit 19 optional features. If any feature is entirely unobserved in a batch, its distribution cannot be estimated; drift evaluation raises an explicit insufficient-data error. It does not fabricate “no drift” or change the 21-feature denominator. The minimal age/sex JSONL case is tested. Use representative feature-complete survey traffic for the current demonstration. Supporting partial coverage in production needs an agreed unavailable/coverage reporting contract; missingness-only alerts would also need a team decision.

## Finish the live handoff

1. **Niranjan:** merge the successful-prediction JSONL writer using the handbook contract, configurable `PREDICTION_LOG_PATH`, request ID, model version and latency; expose/persist logs in the local Compose stack. Never log `seqn`, target labels, credentials or identifiable data. Logging failure must not break prediction serving.
2. **Rishabh, after that integration:** start the local API/Compose stack with logging enabled. In a second terminal, replay `batch_01.csv` at the default 5 requests/s; it takes about 4 minutes 20 seconds. Confirm successful-request log rows appear and invalid requests stay out of the prediction log. This is the remaining strict P2 simulator “done when” item.
3. **Rishabh:** run `python -m src.monitoring.drift --current logs/predictions.jsonl --out reports/drift/live_logs`. Confirm the model version matches and all 21 feature distributions have sufficient observations. Save the demo screenshot. Use the labelled source CSV for the matched performance/trigger workflow; do not pair a JSONL fingerprint with an unrelated CSV performance report.
4. **Rishi:** review/merge PRs 28→32 in dependency order, retargeting bases as needed. Add the handbook's monitoring CI step using the commands below. Any scheduled runs or alert routing need a separate team decision on cadence and ownership; they are not silently assumed to be implemented by the existing data CI.
5. **Rhea/team:** review version 3, the fairness findings and old-population recall tradeoff. Complete model-governance release review. Updating `champion`, exporting the serving artifact and deploying belong to the reviewed release process.

Commands for the workflow owner:

```powershell
python -m scripts.monitor_batch --current data/production/2021-2023_clean.csv --out reports/drift/full_production
python -m src.monitoring.retrain_trigger --drift reports/drift/full_production/drift_summary.json --performance reports/drift/full_production/performance.json --out reports/retraining/trigger.json
```

A valid decision exits 0 whether true or false; the workflow must inspect `retrain_needed`. Missing/malformed inputs exit 2. Retraining is opt-in (`python -m src.models.retrain --trigger reports/retraining/trigger.json --log-mlflow`); no monitoring command promotes the champion automatically. Keep the MLflow credentials in the environment/local ignored `.env`.
