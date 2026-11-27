"""Replay saved evidence at alternative drift thresholds; never change live rules.

Run: python -m src.monitoring.trigger_sensitivity
Only reports are read/written. No monitoring, retraining or deployment is run.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

from src.monitoring.retrain_trigger import RULES, decide

ROOT = Path(__file__).resolve().parents[2]
DRIFT_THRESHOLDS = (0.30, 0.40, 0.50)
BATCHES = tuple(f"batch_{n:02}" for n in range(1, 7)) + (
    "batch_shifted", "full_production", "negative_control")
REASON_CODES = {"data drift": "D", "recall degradation": "R",
                "ROC-AUC below quality-gate floor": "A"}


def analyse_batch(drift, performance):
    """Reuse production validation/metric decisions without mutating RULES."""
    baseline = decide(drift, performance)
    count = drift["feature_count"]
    names = drift["drifted_features"]
    if (type(count) is not int or count < 1 or not isinstance(names, list) or
            any(not isinstance(name, str) or not name for name in names) or
            len(set(names)) != len(names) or len(names) > count):
        raise ValueError("Drift evidence needs a positive feature_count and unique drifted_features")
    share = baseline["values"]["share_drifted_features"]
    if not math.isclose(share, len(names) / count, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Drifted feature count and share disagree")
    # Keep both performance rules exactly as the production trigger evaluated
    # them. Only replace the drift-share comparison in each report scenario.
    performance_reasons = [reason for reason in baseline["reasons"] if reason != "data drift"]
    scenarios = {}
    for threshold in DRIFT_THRESHOLDS:
        reasons = (["data drift"] if share >= threshold else []) + performance_reasons
        scenarios[f"{threshold:.2f}"] = {
            "retrain_needed": bool(reasons), "reasons": reasons,
            "rules": {**RULES, "share_drifted_features": threshold},
        }
    return {"feature_count": count, "drifted_feature_count": len(names),
            "drifted_features": names, "share_drifted_features": share,
            "recall": baseline["values"]["recall"],
            "roc_auc": baseline["values"]["roc_auc"],
            "model_version": baseline["model_version"],
            "data_sha256": baseline["data_sha256"], "scenarios": scenarios}


def read_analysis(directory):
    """Read all nine batches and reject a stale/inconsistent manifest."""
    sources = {}

    def read(relative):
        data = (directory / relative).read_bytes()
        sources[relative] = hashlib.sha256(data).hexdigest()
        return json.loads(data.decode("utf-8"))

    manifest = read("batch_results.json")
    if (not isinstance(manifest, list) or len(manifest) != len(BATCHES) or
            {entry["batch"] for entry in manifest} != set(BATCHES)):
        raise ValueError("batch_results.json must contain each of the nine expected batches once")
    entries = {entry["batch"]: entry for entry in manifest}
    results = []
    for batch in BATCHES:
        drift = read(f"{batch}/drift_summary.json")
        performance = read(f"{batch}/performance.json")
        row = {"batch": batch, **analyse_batch(drift, performance)}
        entry = entries[batch]
        baseline = decide(drift, performance)
        if (type(entry["drifted"]) is not int or entry["drifted"] != row["drifted_feature_count"] or
                entry["n"] != performance["n"] or
                any(not math.isclose(entry[key], row[key], rel_tol=0, abs_tol=1e-12)
                    for key in ("recall", "roc_auc")) or
                type(entry["retrain_needed"]) is not bool or
                entry["retrain_needed"] != baseline["retrain_needed"] or
                entry["reasons"] != baseline["reasons"]):
            raise ValueError(f"batch_results.json disagrees with source reports for {batch}")
        results.append(row)
    return {"date": datetime.now(timezone.utc).isoformat(),
            "analysis_only": True,
            "note": "Counterfactual replay of saved reports; production rules are unchanged. "
                    "No new data/model evaluation or retraining was performed.",
            "production_rules": RULES.copy(), "drift_thresholds": list(DRIFT_THRESHOLDS),
            "source_report_sha256": sources, "batches": results}


def format_table(report):
    """Show every outcome and cause; D=drift, R=recall, A=ROC-AUC."""
    rows = [["Batch", "Drifted", "Share", "Recall", "ROC-AUC", "0.30", "0.40", "0.50"]]
    for batch in report["batches"]:
        decisions = []
        for threshold in DRIFT_THRESHOLDS:
            scenario = batch["scenarios"][f"{threshold:.2f}"]
            causes = "+".join(REASON_CODES[reason] for reason in scenario["reasons"])
            decisions.append(f"YES:{causes}" if scenario["retrain_needed"] else "NO")
        rows.append([batch["batch"], f"{batch['drifted_feature_count']}/{batch['feature_count']}",
                     f"{batch['share_drifted_features']:.4f}", f"{batch['recall']:.4f}",
                     f"{batch['roc_auc']:.4f}", *decisions])
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(value.ljust(width) for value, width in zip(row, widths)) for row in rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports/drift")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/retraining/trigger_sensitivity.json")
    args = parser.parse_args(argv)
    try:
        report = read_analysis(args.reports_dir)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        args.out.unlink(missing_ok=True)
        print(f"Cannot analyse trigger sensitivity ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    print("REPORT ONLY - production drift/recall/ROC-AUC trigger rules are unchanged")
    print(format_table(report))
    print("D = data drift; R = recall degradation; A = ROC-AUC below quality-gate floor")
    print(f"Sensitivity report saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
