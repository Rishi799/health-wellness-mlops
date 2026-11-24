"""Consolidate saved holdout evidence for issue #25 without refitting a model.

Run: python -m src.monitoring.fairness_followup
This audits existing reports, not raw predictions or a live registry. It never
changes thresholds, aliases, data, or serving artifacts.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[2]
CYCLES = {"2017-2018": "2017", "2021-2023": "2021"}
GROUPS = {"sex": ("1", "2"), "age": ("20-39", "40-59", "60-79", "80")}
TEST_PATHS = {"2017-2018": "data/processed/test.csv",
              "2021-2023": "data/production/test_2021.csv"}


def wilson_interval(detected, positives):
    """Descriptive binomial interval; not adjusted for NHANES survey design."""
    if not positives:
        return None
    z = NormalDist().inv_cdf(0.975)
    p = detected / positives
    denominator = 1 + z * z / positives
    center = (p + z * z / (2 * positives)) / denominator
    radius = z * math.sqrt(p * (1 - p) / positives + z * z / (4 * positives ** 2)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _same(value, expected):
    return (value is None and expected is None) or (
        isinstance(value, (int, float)) and not isinstance(value, bool) and
        expected is not None and math.isfinite(value) and
        math.isclose(value, expected, abs_tol=1e-10))


def build_followup(comparison, assessments, champion_info):
    """Check provenance and counts before joining the four holdout assessments."""
    rows = []
    model_hashes = {}
    for model in ("champion", "challenger"):
        for cycle in CYCLES:
            report = assessments[(model, cycle)]
            expected = comparison[model][cycle]
            source = (champion_info["source_run_id"] if model == "champion"
                      else comparison["source_run_id"])
            if (not source or report["source_run_id"] != source or
                    report["data_sha256"] != comparison["input_sha256"][TEST_PATHS[cycle]] or
                    report["threshold"] != champion_info["decision_threshold"] or
                    expected["decision_threshold"] != report["threshold"]):
                raise ValueError(f"Mismatched run, holdout or threshold: {model} {cycle}")
            artifact = report["model_sha256"]
            if (not isinstance(artifact, str) or len(artifact) != 64 or
                    (model in model_hashes and model_hashes[model] != artifact) or
                    (model == "challenger" and artifact != comparison["artifact_sha256"])):
                raise ValueError(f"Mismatched model fingerprint: {model} {cycle}")
            model_hashes[model] = artifact
            if (report["n"] != expected["n"] or
                    report["overall"]["positives"] != expected["positives"]):
                raise ValueError(f"Mismatched overall counts: {model} {cycle}")
            for field, names in GROUPS.items():
                groups = report[field]["groups"]
                if len(groups) != len(names) or {g["group"] for g in groups} != set(names):
                    raise ValueError(f"Missing or duplicate groups: {model} {cycle} {field}")
                if (sum(g["n"] for g in groups) != expected["n"] or
                        sum(g["positives"] for g in groups) != expected["positives"]):
                    raise ValueError(f"Subgroup totals differ from holdout: {model} {cycle}")
                for name in names:
                    group = next(g for g in groups if g["group"] == name)
                    n, positives, detected = (group[k] for k in ("n", "positives", "detected"))
                    if (any(type(v) is not int for v in (n, positives, detected)) or
                            not 0 <= detected <= positives <= n):
                        raise ValueError("Invalid subgroup counts")
                    recall = detected / positives if positives else None
                    matching = [g for g in expected["subgroups"]
                                if g["field"] == field and g["group"] == name]
                    if (len(matching) != 1 or matching[0]["n"] != n or
                            matching[0]["positives"] != positives or
                            not _same(matching[0]["recall"], recall) or
                            not _same(group["recall"], recall)):
                        raise ValueError(f"Subgroup counts/recall disagree: {model} {cycle} {name}")
                    interval = wilson_interval(detected, positives)
                    saved = group["recall_ci95_wilson"]
                    if ((interval is None and saved is not None) or
                            (interval is not None and
                             (not isinstance(saved, list) or len(saved) != 2 or
                              not all(_same(a, b) for a, b in zip(saved, interval))))):
                        raise ValueError("Saved uncertainty interval disagrees with counts")
                    rows.append({"model": model, "cycle": cycle, "field": field,
                                 "group": name, "n": n, "positives": positives,
                                 "detected": detected, "missed": positives - detected,
                                 "recall": recall, "recall_ci95_wilson": interval})
    return {
        "date": datetime.now(timezone.utc).isoformat(),
        "evidence_type": "consistency-checked saved aggregate reports; no fresh model evaluation",
        "issue": "https://github.com/Rishi799/health-wellness-mlops/issues/25",
        "decision_threshold": champion_info["decision_threshold"],
        "source_runs": {"champion": champion_info["source_run_id"],
                        "challenger": comparison["source_run_id"]},
        "model_sha256": model_hashes,
        "rows": rows,
        "uncertainty_note": "Descriptive 95% Wilson binomial intervals; small positive counts, "
                            "no survey weights/design correction, no causal or fairness proof.",
        "release_status": "Human review required; no champion, threshold or deployment change.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository evidence directory")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/retraining/fairness.json")
    args = parser.parse_args(argv)
    sources = {}

    def read(relative):
        data = (args.root / relative).read_bytes()
        sources[relative] = hashlib.sha256(data).hexdigest()
        return json.loads(data.decode("utf-8"))

    try:
        comparison = read("reports/retraining/comparison.json")
        champion_info = read("models/model_info.json")
        assessments = {(model, cycle): read(f"reports/fairness/{model}_{suffix}.json")
                       for model in ("champion", "challenger") for cycle, suffix in CYCLES.items()}
        result = build_followup(comparison, assessments, champion_info)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Do not leave a stale successful summary after a failed rerun.
        args.out.unlink(missing_ok=True)
        print(f"Fairness evidence rejected: {type(exc).__name__}: {exc}")
        return 1
    result["source_report_sha256"] = sources
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print("Model      Holdout    Field Group  n  Detected/positives  Recall")
    for row in result["rows"]:
        recall = "unavailable" if row["recall"] is None else f"{row['recall']:.4f}"
        print(f"{row['model']:<10} {row['cycle']} {row['field']:<5} {row['group']:<5} "
              f"{row['n']:4} {row['detected']}/{row['positives']} {recall}")
    print(f"Saved evidence summary to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
