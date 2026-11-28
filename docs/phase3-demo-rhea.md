# Rhea's Phase 3 demo

Allow three to four minutes for this segment, following Rishabh's trigger demo
and handing over to Rishi. This is a saved-evidence review and report-only
sensitivity replay, not a live model promotion or retraining demonstration.
Keep the [comparison](../reports/retraining/comparison.json),
[numerical decision](../reports/retraining/promotion_decision.json),
[governance decision](governance.md#release-decision--challenger-v3),
[sensitivity report](../reports/retraining/trigger_sensitivity.json),
[model card](model_card.md) and [fairness issue #25](https://github.com/Rishi799/health-wellness-mlops/issues/25)
open before starting. Match Rishabh's [demo](phase3-demo-rishabh.md) and identify
all results as saved evidence from the report files.

## Preflight

- Open this repository's root in VS Code and a PowerShell terminal. Use the
  existing Python 3.11 `.venv`; do not install packages during the presentation.
- The commands below need committed JSON reports, not Docker, a live API,
  DVC downloads or DagsHub sign-in. Avoid `--register`, model export, alias writes
  and deployment commands. Do not run `compare_champion` merely to display its
  saved results: that would require candidate/data artifacts and rewrite evidence.
- Run the sensitivity command before presenting, keep its terminal output
  visible, and scroll to it as a backup. Finish tests before rehearsal.
- Ask the team to fill the approval names/date in the governance document when
  actually approved. Until then, say the keep-v2 decision is recorded for sign-off;
  never present placeholders as proof of completed approval.

```powershell
.\.venv\Scripts\python.exe -m src.monitoring.trigger_sensitivity
.\.venv\Scripts\python.exe -m pytest tests/test_trigger_sensitivity.py -v
```

## Four-minute script

| Time | Show and say |
| --- | --- |
| 0:00-0:45 | Show the saved comparison using the command below. "We compare both models on the same two holdouts. Later-holdout ROC-AUC rises from 0.8401 to 0.8457, but recall stays at 0.8191. On the original holdout, candidate recall falls from 0.8270 to 0.8062. A newer model is not automatically a better release." |
| 0:45-1:15 | Open `promotion_decision.json`. "The fixed rule requires passing hard quality checks, no later-recall decrease, later AUC within 0.01 of the champion, and original AUC at least 0.80. The saved result is numerically eligible: `promote: true`, but `review_required: true` and `champion_alias_changed: false`. Eligibility is separate from release approval." |
| 1:15-1:55 | Show the governance release decision and fairness counts. "The recorded team choice is to keep v2 as champion and v3 as challenger. Later recall is equal, the AUC gain is modest, and original recall declines. Young-adult detection only changes from 1/14 to 3/14 on the later holdout, so the candidate still misses 11 positives. The approval line must be completed by the team; this demo changes no alias or serving model." |
| 1:55-2:55 | Run the sensitivity command or show its prepared output. "We replay the existing evidence at 30%, 40% and 50% drift-share cutoffs, keeping both performance rules fixed. At 30%, every ordinary batch triggers on drift. At 50%, only the synthetic shifted batch triggers on drift, but batches 03, 04 and 06 still trigger on recall. The identity control remains negative. Considering 50% is a recommendation, not an implemented rule change; a higher cutoff may miss earlier warning signs." |
| 2:55-3:35 | Open the model card and governance threshold review. "The model card records intended use, calibration and subgroup limitations. Governance records who approves changes, versioning and rollback. We propose considering a higher drift-share cutoff only through team review; the operational rule still uses 0.30, recall below 0.777 or AUC below 0.80." |
| 3:35-4:00 | Hand over: "Our automated checks produce evidence, while the human release decision considers tradeoffs. The champion remains v2 under the recorded decision, and no rule was changed by the sensitivity analysis. Rishi will close with integration, remaining approvals and the final project summary." |

## Exact display commands

Run from the repository root in PowerShell. These commands display existing
reports; only the sensitivity command regenerates its own analysis JSON.

```powershell
# Compact saved champion/challenger comparison; no fit or evaluation is run.
$comparison = Get-Content reports/retraining/comparison.json -Raw | ConvertFrom-Json
foreach ($model in 'champion', 'challenger') {
    foreach ($cycle in '2017-2018', '2021-2023') {
        $scores = $comparison.$model.$cycle
        '{0} | {1} | ROC-AUC {2:F4} | recall {3:F4}' -f $model, $cycle, $scores.roc_auc, $scores.recall
    }
}

# Distinguish numerical eligibility from the human release decision.
Get-Content reports/retraining/promotion_decision.json
Get-Content docs/governance.md

# Report-only scenarios: does not edit production rules or run retraining.
.\.venv\Scripts\python.exe -m src.monitoring.trigger_sensitivity
$LASTEXITCODE

# Show the full-precision saved analysis and model limitations if asked.
Get-Content reports/retraining/trigger_sensitivity.json
Get-Content docs/model_card.md
```

Exit 0 from sensitivity means the analysis was generated successfully, not
"no batch needs retraining." The scenario cells contain those decisions.
Its readable table uses D=input drift, R=recall degradation and A=low ROC-AUC.

## Faculty questions

| Question | Short answer |
| --- | --- |
| Why keep v2 when the decision says `promote: true`? | That flag means the minimum numerical conditions passed. The team can still retain v2 because original recall decreases, later recall is equal and subgroup risks persist. Approval placeholders are not completed approval evidence. |
| Why is an exact recall tie eligible? | The prescribed rule uses >=, so equal later recall satisfies it. We did not change that rule; eligibility does not compel release. |
| Why compare both cycles? | We check adaptation to later data while retaining an original-population floor. Both holdouts were excluded from fitting and preprocessing; inspected holdouts must not become tuning data. |
| Did you change 0.30 to 0.50? | No. I replayed saved reports at three drift cutoffs. Production still uses 0.30 and the same recall/AUC rules. A code change would require a separate reviewed team decision. |
| Why consider 0.50? | In these reports it suppresses ordinary drift-only requests but retains the shifted-batch drift alert and the independent performance alerts. That is an operational tradeoff, not proof of an optimal threshold. |
| What is the risk of raising the cutoff? | It may delay or miss meaningful input shifts, especially before labels arrive. Ordinary cohort drift is not proven a false alarm, and one synthetic stress case cannot represent all future shifts. |
| What do 0.30, 0.40 and 0.50 measure? | The fraction of the 21 features flagged as drifting, not a changed per-feature distance cutoff and not the model's classification threshold. |
| Why do batches 03, 04 and 06 still trigger at 0.50? | Their recall is below 0.777. The trigger combines drift, recall and AUC with OR, so any condition is sufficient. |
| Did retraining fix the young-adult issue? | No. Later detection increases from 1/14 to 3/14 but remains poor and uncertain; the candidate's Wilson recall interval is 0.0757-0.4759. See issue #25 and the fairness reports. |
| Can this replay establish clinical safety? | No. These are engineering criteria on an educational self-reported-diagnosis dataset, with calibration and subgroup limitations. The service is not a diagnostic tool. |
| Is the full cohort another independent batch? | No. Full production aggregates the later cohort; its rows overlap the ordinary batches. Neither it nor these scenarios should be counted as independent validation experiments. |

## Backup plan

- **Terminal/Python unavailable:** show the committed sensitivity JSON and the
  same table in governance. Explain that these are results from the saved replay.
- **GitHub/DagsHub unavailable:** use local comparison, decision, fairness and
  model-card files; no remote access is needed to explain the evidence.
- **Missing or inconsistent reports:** show the command's clear error, use the
  committed review copy, and investigate afterwards. Do not edit values, suppress
  alerts or treat an old output as a newly successful run.
- **Short on time:** show the four comparison rows, eligibility versus keep-v2
  decision, and the 0.30/0.50 contrast; point to the model card and hand to Rishi.
- **Asked to promote live:** explain that alias/export/deployment changes are a
  reviewed release operation, not part of this read-only demonstration.
