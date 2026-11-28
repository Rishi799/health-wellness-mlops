# Model governance and release review

Owner: Rhea (P3). Prepared 22 November 2026 against main commit
`45bdab4`. This document defines the proposed review procedure for the team's
educational system. A merged implementation or numerical PASS is not clinical
approval. Teammate review of this document is still required.

## Purpose and boundaries

The platform provides awareness and education for adults, using a label of prior
self-reported cardiometabolic diagnosis. It must not diagnose disease, predict
an individual's future event risk, recommend treatment, triage patients, or
determine access to care, insurance, employment, or other services. Use synthetic
or public de-identified NHANES records in demonstrations. A low score is not a
reason to dismiss symptoms or delay professional care.

## Owners and decision rights

- **Rishi (P1), coordinator:** shared contracts, integration, PR review/merge,
  CI/CD coordination and final report. Coordinates the human release decision.
- **Rishabh (P2), data and drift:** cleaning/versioning, production batches,
  simulation, drift/performance evidence and retraining trigger.
- **Rhea (P3), model quality and governance:** retraining/comparison review,
  fixed quality criteria, model card, subgroup follow-up and release risks.
- **Niranjan (P4), serving and observability:** API logging, metrics, dashboard,
  monitoring stack and observability runbook. Deployment access is exercised by
  the authorized Render/GitHub maintainers under the existing release runbook.

The PR author does not approve their own change. Follow the handbook's minimum
of one teammate approval and green required CI checks. A model replacement also
needs an explicit team decision recorded in the release PR; an approval of
retraining code alone does not authorize replacing the served champion.

## Lifecycle and approval flow

1. **Observe:** P2 evaluates a labelled production batch and publishes matching
   drift/performance reports with data and model fingerprints. Do not combine
   reports from different batches or models.
2. **Trigger:** the existing rules request retraining if drifting input share
   is at least 0.30, recall is below 0.777, or ROC-AUC is below 0.80. These are
   project rules, not universal statistical or clinical standards. Preserve the
   actual outcome, including ordinary batches that drift.
3. **Fit a separate candidate:** `src/models/retrain.py` requires a true trigger
   or explicit recorded `--force` experiment. Split the later data 80/20 with
   stratification and seed 42 before fitting. Only its training portion joins
   the original training set. Reuse the same 21-feature contract, preprocessing
   builder and Logistic Regression configuration. Never train on either test
   set; participant-overlap assertions and provenance hashes are required.
4. **Evaluate:** compare both models on the same original and later holdouts.
   Import the existing quality gate without changing its rules. Save comparison,
   quality results, subgroup counts, calibration and decision evidence.
5. **Determine numerical eligibility:** all hard gate checks must pass; later
   recall must not decrease; later ROC-AUC must be at least champion minus 0.01;
   original ROC-AUC must remain at least 0.80. Exact ties are eligible under the
   current rule. `promote: true` means eligible for review, not deployed.
6. **Register for review:** the opt-in `--register` path may create/reuse a
   candidate version and assign alias `challenger`. It must not move `champion`.
7. **Human release review:** the PR records run/version, artifact/data hashes,
   all four model/holdout results, fairness/calibration warnings, the original
   population tradeoff, and the team's accept/defer/reject rationale. Rhea
   reviews model risks; Rishi coordinates the teammate approval. Do not edit
   eligibility numbers to manufacture approval.
8. **Approved release only:** authorized maintainers move the champion alias
   and export the approved model/metadata as a consistent pair using the
   existing registry/export workflow. DVC versions the artifact. The reviewed
   release PR must pass CI before merge. Confirm the source run and version in
   the exported metadata. Never switch the alias with an unreviewed candidate.
9. **Deliver and verify:** successful trusted-main CI enables CD to build,
   smoke-test and publish the image, then deploy when Render is configured.
   Verify the public service's application commit, model version and prediction
   response. A hook acknowledgement, healthy old instance or registry alias
   change does not prove the intended model is deployed.
10. **Observe or roll back:** keep the previous verified image digest and model
    artifacts. Record the release result and any incident before further changes.

The immutable source for quality checks is [quality_gate.py](../src/models/quality_gate.py);
the comparison rule is [compare_champion.py](../src/models/compare_champion.py).
Repeated evaluation of an already-inspected holdout is a regression/audit check,
not a new unbiased selection experiment. Any tuning uses training-only validation.

## Current candidate and unresolved risks

The saved [comparison](../reports/retraining/comparison.json) and
[decision](../reports/retraining/promotion_decision.json) identify candidate
run `893b1b05690d4870ba99b0628355395f`. The decision records registry version 3
with alias `challenger`, `review_required: true` and
`champion_alias_changed: false`. The [remote audit](../reports/phase3/remote_verification.json)
recorded champion version 2 on 20 November; that is historical evidence, not a
fresh assertion about today's registry or deployment.

The candidate meets the numerical rule but the young-adult weakness persists,
and original-holdout recall declines. These are explicit human-review items in
the [model card](model_card.md) and [issue #25](https://github.com/Rishi799/health-wellness-mlops/issues/25).
Rhea recommends keeping the current serving artifact unchanged while the team
reviews those tradeoffs. This recommendation does not change the eligibility
formula, register a new model, or claim a team decision has already happened.

## Release decision — challenger v3

**Decision recorded:** 27 November 2026.

**Team decision:** keep version 2 as `champion` and leave version 3 as
`challenger`. This records the team's release choice; it does not execute a
registry, serving-artifact or deployment change. It supersedes the earlier
recommendation to wait for a release decision.

**Approved by:** Rishi Agarwal (P1), Rishabh (P2), Rhea Parthiban (P3) and
Niranjan (P4) on 28 November 2026.

All four team members confirmed both points: (1) keep version 2 as `champion`
and leave version 3 as `challenger`; (2) treat a 0.50 drift-share cutoff as the
recommendation from the [trigger threshold review](#trigger-threshold-review),
while the implemented trigger rule stays at 0.30 until a separate reviewed
change.

The [saved comparison](../reports/retraining/comparison.json) supports retaining
v2 even though the [numerical decision](../reports/retraining/promotion_decision.json)
reports `promote: true`:

- Original-holdout recall decreases from **0.8270** for v2 to **0.8062** for v3.
- Later-holdout recall is equal at **0.8191**; the priority metric does not improve.
- Later-holdout ROC-AUC improves only from **0.8401** to **0.8457**. This modest
  ranking gain does not by itself justify replacing the champion.
- Young-adult detection remains poor: **2/17 versus 3/17** on the original
  holdout, and **1/14 versus 3/14** on the later holdout. See the
  [four-model/holdout fairness summary](../reports/retraining/fairness.json),
  [source Fairlearn reports](../reports/fairness/) and
  [issue #25](https://github.com/Rishi799/health-wellness-mlops/issues/25).

Numerical eligibility is a minimum bar for human review, not an obligation to
release. The existing gate permits this candidate, while the team considers
old-population recall, equal later recall and unresolved subgroup risks in its
release choice. The saved eligibility JSON is intentionally unchanged. Any
future replacement needs new reviewed evidence and explicit release approval.

## Trigger threshold review

**Review date:** 27 November 2026. **Status:** recommendation for team decision;
not an implemented change. Production remains drift share **>= 0.30**, recall
**< 0.777**, or ROC-AUC **< 0.80**, combined with OR.

The [sensitivity report](../reports/retraining/trigger_sensitivity.json) reads
[batch_results.json](../reports/drift/batch_results.json) and each batch's
`drift_summary.json` and `performance.json`. It checks their consistency and
replays the same labelled evidence with drift-share cutoffs 0.30, 0.40 and 0.50;
both performance cutoffs remain fixed. No new predictions, retraining, rule
edits or threshold optimization are performed. Values below are rounded to
four decimals; the JSON retains full precision and source-report hashes.

Each scenario cell answers **whether any rule requests retraining**, followed
by its causes: **D** = input drift, **R** = recall below 0.777, **A** = ROC-AUC
below 0.80. A drift share equal to the scenario cutoff triggers; performance
values equal to their floors do not.

| Batch | Drifted features | Share | Recall | ROC-AUC | Drift cutoff 0.30 | Drift cutoff 0.40 | Drift cutoff 0.50 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| batch_01 | 10/21 | 0.4762 | 0.7934 | 0.8280 | Yes: D | Yes: D | No |
| batch_02 | 8/21 | 0.3810 | 0.8100 | 0.8476 | Yes: D | No | No |
| batch_03 | 8/21 | 0.3810 | 0.7741 | 0.8156 | Yes: D+R | Yes: R | Yes: R |
| batch_04 | 9/21 | 0.4286 | 0.7558 | 0.8083 | Yes: D+R | Yes: D+R | Yes: R |
| batch_05 | 9/21 | 0.4286 | 0.8149 | 0.8259 | Yes: D | Yes: D | No |
| batch_06 | 9/21 | 0.4286 | 0.7588 | 0.8242 | Yes: D+R | Yes: D+R | Yes: R |
| batch_shifted | 13/21 | 0.6190 | 0.9293 | 0.7533 | Yes: D+A | Yes: D+A | Yes: D+A |
| full_production | 8/21 | 0.3810 | 0.7846 | 0.8248 | Yes: D | No | No |
| negative_control | 0/21 | 0.0000 | 0.8270 | 0.8502 | No | No | No |

At 0.30, all six ordinary later-cycle batches and the full production cohort
request retraining on drift; the synthetic shifted batch also triggers on
drift and low ROC-AUC. At 0.50, only that synthetic batch crosses the drift
cutoff, while batches 03, 04 and 06 still request retraining through recall.
The negative control never triggers. These are saved correlated batches from
one survey cohort, plus a full-cohort aggregation and controls, not independent
production weeks proving an optimal threshold.

**Recommendation:** consider a drift-share cutoff of **0.50** for team review,
while keeping the recall and ROC-AUC conditions unchanged. On these reports it
reduces drift-only retraining requests while preserving the measured
performance alerts. The tradeoff is delayed or missed input-drift alerts when
labelled performance is unavailable or degradation has not yet appeared;
ordinary-cohort drift must not be re-labelled a false alarm solely because a
higher cutoff suppresses it. A single synthetic stress batch does not establish
sensitivity to all future shifts. Decide using monitoring costs, label delay,
measurement changes and further representative evidence before proposing a
separate reviewed code change. The live rule remains 0.30 until that decision.

Reproduce this report-only analysis from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.monitoring.trigger_sensitivity
```

## What is versioned where

- **Git/GitHub:** source, tests, workflow definitions, contracts, model metadata,
  DVC pointer/lock files, aggregate evidence, model card, decisions and reviews.
- **DVC/DagsHub storage:** raw/clean data, original/later partitions and model
  bytes. Never commit CSV/XPT records, candidate artifacts or caches to Git.
- **MLflow/DagsHub:** experiment runs, configuration, metrics, artifact source,
  registered versions and aliases. Run IDs connect comparison to registration.
- **GHCR:** tested container images tagged by application commit SHA. Record
  the image digest too; retain previous verified digests for rollback rather
  than relying on a mutable `latest` tag.
- **Render:** deployment history and running service. The baked model bytes
  determine the served model; moving a registry alias alone cannot update them.

## Audit trail and evidence retention

Every candidate should connect trigger reasons and source reports to data hashes,
training configuration, Git commit, MLflow run, comparison, review decision and,
if released, DVC model version, image digest, CI/CD runs and public smoke result.
Keep aggregate reports and release/rollback records with the project history.
Never publish per-person feature vectors, credentials or local private paths in
those reports. The fairness summaries contain counts and aggregate metrics only.

Prediction logs are operational evidence, not labels or automatically suitable
training data. A real deployment needs trustworthy outcome linkage and consent/
privacy arrangements before using such data for evaluation or training. The
current educational replay instead uses separately labelled public survey CSVs.

## Privacy, access and operational logging

The handbook log contract specifies timestamp, random request ID, model version,
the 21 serving inputs (null when omitted), probability, score, band and latency.
Successful predictions belong in JSONL; errors belong in the ordinary error log.
Do not log names, `seqn`, target/diagnosis component columns, credentials, IPs,
or identifiers that could link a record to a person. A random request ID is for
operational correlation; it must not embed a participant identifier.

Even without names, combinations of health and demographic inputs may be
sensitive. Restrict runtime logs to authorized team operators, keep them outside
Git and Docker build context, and share aggregate reports in reviews. Tokens
remain in ignored local environment files or platform secret stores; never
paste them into issues, screenshots or reports.

**Implementation boundary:** current main has no successful-prediction JSONL
writer, `/metrics`, monitoring stack or log rotation/retention implementation.
Niranjan's work must be integrated and verified before claiming a live feedback
loop. Git-ignore/docker-ignore rules alone are not retention or access controls.

**Team decisions still needed before collecting real user health inputs:**
Niranjan and Rishi must agree and document a concrete retention duration, storage
location/persistence, operator access, deletion/rotation procedure and collection
notice. No duration is invented here or claimed to be enforced. Until those
controls are reviewed, use synthetic/public survey demo traffic only. Render's
ephemeral filesystem is not a durable audit store and does not enforce a known
retention period. Monitoring cadence, alert destination and acknowledgement
owner also require explicit configuration; no schedule is assumed to exist.

## Incident response and rollback

1. Record the symptom and current application SHA, image digest, model version,
   time and affected endpoint. Preserve sanitized aggregate evidence. Notify
   Rishi and the relevant owner; do not publish sensitive request bodies.
2. For startup/serving defects or an unacceptable model-release outcome, the
   authorized maintainer selects the previous verified image/deploy and follows
   [the deployment and rollback runbook](render-deployment.md#5-rehearse-image-rollback).
3. Verify Live status and run the documented smoke test against the expected
   previous application SHA. Check model version, serving contract and one
   synthetic prediction. Preserve the previous and restored image digests.
4. Align the intended registry alias and release metadata through a reviewed
   follow-up; rolling back an image alone does not update MLflow aliases.
5. Record the reason, reviewer, previous/restored identities, verification and
   remediation. Re-release only after the corrected change passes review/CI.

The runbook includes a recorded Phase 2 rollback-and-restore drill on
10 November 2026. It is prior evidence; no new rollback was performed for this
governance update.

## Change control and acceptance checklist

Changing feature/label definitions, trigger thresholds, hard versus warning
severity, promotion tolerances, classification threshold or display bands needs
Rishi's coordination and a recorded team decision before implementation. Include
the rationale, training-only validation, subgroup/calibration impact, tests and
review. Never tune on a test set or lower thresholds merely to obtain PASS.

Before closing Rhea's governance work, a teammate must review the model card,
this policy and the fairness interpretation. Keep issue #25 open while its
investigation items remain unresolved. Before a model release, additionally
record explicit team approval, green CI, version/image provenance, the verified
deployment and a viable rollback target. Pending teammate actions are not
silently represented as completed by this document.
