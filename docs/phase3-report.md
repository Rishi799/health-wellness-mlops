# Phase 3 implementation report

## Rishabh — production data, monitoring, trigger and challenger handoff

The later NHANES cycle produces 7,783 validated adult records with the original 31-column contract. Seed-42 partitioning gives six disjoint simulated production batches; a separate, explicitly synthetic older-adult batch adds clipped BMI/SBP offsets. A full DVC rebuild preserved all seven original-cycle CSV fingerprints. Data, splits and the candidate artifact are uploaded through DVC.

The traffic simulator sends serving features only, respects a configurable rate and reports HTTP/transport outcomes. A bounded local replay returned 20/20 successful predictions; another replay returned two successes and two expected validation failures.

Evidently found input drift in 8/21 features across the full later cycle. Its ROC-AUC was 0.8248 and recall 0.7846. The three-rule trigger therefore requested retraining for data drift. The synthetic stress batch triggered drift and the ROC-AUC floor; an identical-data negative control requested no retraining. Reports preserve ordinary-batch drift rather than changing the rule to force an expected demonstration.

The challenger fits 10,654 rows, excluding both holdouts. On the later holdout it achieves ROC-AUC 0.8457 versus champion 0.8401, with equal recall 0.8191. On the original holdout it achieves ROC-AUC 0.8495 and recall 0.8062. All prescribed numerical promotion checks pass. The candidate is registered as version 3 with alias `challenger`; the `champion` alias remains version 2. Updating the serving model requires the team's reviewed release process.

Fairlearn shows that retraining does not resolve young-adult performance: the challenger detects 3/14 positive cases aged 20–39 on the later holdout, compared with 1/14 for the champion. Small positive counts make the estimate uncertain. Original-population recall also decreases. These limitations remain explicit release-review items. SHAP explains the fitted Logistic Regression in log-odds, with contributions aggregated to the 21 inputs and reconstruction error around `1e-15`; the attributions do not establish causal health effects.

The [successful DagsHub run](https://dagshub.com/Rishi799/health-wellness-mlops.mlflow/#/experiments/2/runs/893b1b05690d4870ba99b0628355395f) contains candidate parameters, both holdout metrics, model artifact and aggregate review evidence. Git tracks code, tests, DVC pointers and summary reports. MLflow tracks the run/registry; DVC tracks data/model bytes; GHCR and Render retain their Phase 2 release roles. The [runbook](phase3-p2.md) provides commands, exact results and source references.

Delivery is stacked across PRs [28](https://github.com/Rishi799/health-wellness-mlops/pull/28), [29](https://github.com/Rishi799/health-wellness-mlops/pull/29), [30](https://github.com/Rishi799/health-wellness-mlops/pull/30), [31](https://github.com/Rishi799/health-wellness-mlops/pull/31), and [32](https://github.com/Rishi799/health-wellness-mlops/pull/32). Team review/merges, Niranjan's live JSONL and dashboard integration, Rishi's scheduled workflow/alerts and Rhea's model-release review remain separate integration steps. This report does not claim the whole team's Phase 3 is complete.
