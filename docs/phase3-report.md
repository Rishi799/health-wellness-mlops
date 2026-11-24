# Phase 3 implementation report

## Rhea (P3) - model review, governance and fairness

The existing retraining and champion-comparison implementation was merged in
[PR #37](https://github.com/Rishi799/health-wellness-mlops/pull/37). This review
builds on that work rather than refitting or registering a duplicate candidate.
The [training record](../reports/retraining/training.json) uses 4,428 original
and 6,226 later-cycle fitting rows; the 1,108 original and 1,557 later holdout
rows are excluded. Imported preprocessing/Logistic Regression settings, split
seed 42, participant-separation checks and input/artifact fingerprints connect
the candidate to its evaluation evidence.

The [comparison](../reports/retraining/comparison.json) gives later-holdout
ROC-AUC 0.8457 for the challenger versus 0.8401 for the champion, with equal
recall 0.8191. Challenger original-holdout ROC-AUC is 0.8495 and recall 0.8062,
compared with champion recall 0.8270. The [decision](../reports/retraining/promotion_decision.json)
passes the fixed numerical eligibility rule, registers version 3 as
`challenger`, requires review and records no champion-alias change. The saved
[remote audit](../reports/phase3/remote_verification.json) records version 2 as
champion; this is dated evidence, not a fresh remote check.

Rhea's recommendation is to retain the exported champion while the team reviews
the original-population recall loss and unresolved subgroup risks. Young-adult
detection on the later holdout improves from 1/14 to 3/14; it is still poor and
uncertain. The [model card](model_card.md) links the underlying Fairlearn and
comparison evidence and the open investigation in
[issue #25](https://github.com/Rishi799/health-wellness-mlops/issues/25).

The new [governance document](governance.md) assigns owners and separates
automated eligibility from team release approval. It covers Git/DVC/MLflow/GHCR
versioning, audit evidence, data privacy, pending retention/access decisions,
threshold change control and the existing verified rollback procedure. Live
prediction logging, metrics, dashboards and scheduled alerting must not be
claimed from offline reports; their integration remains with the named owners.
Teammate review of the card/policy and a team decision on any model replacement
remain required.

## Rishabh — production data, monitoring, trigger and challenger handoff

The later NHANES cycle produces 7,783 validated adult records with the original 31-column contract. Seed-42 partitioning gives six disjoint simulated production batches; a separate, explicitly synthetic older-adult batch adds clipped BMI/SBP offsets. A full DVC rebuild preserved all seven original-cycle CSV fingerprints. Data, splits and the candidate artifact are uploaded through DVC.

The traffic simulator sends serving features only, respects a configurable rate and reports HTTP/transport outcomes. The 21 November audit validated all 9,081 batch/stress payloads and replayed all 1,298 rows of the first batch locally: 1,298 HTTP 200 responses, followed by five expected 422s for a separate deliberately invalid run. The API currently lacks the prediction JSONL writer, so the strict traffic → log → live drift acceptance check remains pending Niranjan's integration and a subsequent Rishabh replay.

Evidently found input drift in 8/21 features across the full later cycle. Its ROC-AUC was 0.8248 and recall 0.7846. The three-rule trigger therefore requested retraining for data drift. The synthetic stress batch triggered drift and the ROC-AUC floor; an identical-data negative control requested no retraining. Reports preserve ordinary-batch drift rather than changing the rule to force an expected demonstration.

The challenger fits 10,654 rows, excluding both holdouts. On the later holdout it achieves ROC-AUC 0.8457 versus champion 0.8401, with equal recall 0.8191. On the original holdout it achieves ROC-AUC 0.8495 and recall 0.8062. All prescribed numerical promotion checks pass. The candidate is registered as version 3 with alias `challenger`; the `champion` alias remains version 2. Updating the serving model requires the team's reviewed release process.

Fairlearn shows that retraining does not resolve young-adult performance: the challenger detects 3/14 positive cases aged 20–39 on the later holdout, compared with 1/14 for the champion. Small positive counts make the estimate uncertain. Original-population recall also decreases. These limitations remain explicit release-review items. SHAP explains the fitted Logistic Regression in log-odds, with contributions aggregated to the 21 inputs and reconstruction error around `1e-15`; the attributions do not establish causal health effects.

The [successful DagsHub run](https://dagshub.com/Rishi799/health-wellness-mlops.mlflow/#/experiments/2/runs/893b1b05690d4870ba99b0628355395f) contains candidate parameters, both holdout metrics, model artifact and aggregate review evidence. Git tracks code, tests, DVC pointers and summary reports. MLflow tracks the run/registry; DVC tracks data/model bytes; GHCR and Render retain their Phase 2 release roles. The [runbook](phase3-p2.md) provides commands, exact results and source references.

Delivery is stacked across PRs [28](https://github.com/Rishi799/health-wellness-mlops/pull/28), [29](https://github.com/Rishi799/health-wellness-mlops/pull/29), [30](https://github.com/Rishi799/health-wellness-mlops/pull/30), [31](https://github.com/Rishi799/health-wellness-mlops/pull/31), and [32](https://github.com/Rishi799/health-wellness-mlops/pull/32). Team review/merges, Niranjan's live JSONL and dashboard integration, Rishi's monitoring CI step and Rhea's model-release review remain separate integration steps. This report does not claim the whole team's Phase 3 is complete.

The [completion audit](phase3-verification.md) maps every handbook requirement to evidence and pending owners. Independent recalculation reproduced the reported metrics exactly and verified data/model hashes, split reproduction, participant separation and Fairlearn/SHAP provenance. The audit also fixed stale report outputs after failed monitoring reruns. All **256 local tests** pass; DVC local/remote status and dependency consistency checks pass. Both trigger outcomes and the Evidently report were captured in [screenshots](../reports/phase3/screenshots/), and the full champion degradation table includes precision, Brier and subgroup recall. The ordinary-batch drift differs from the handbook's initial hoped-for story; it is retained as an observed finding for review.
