"""Compare an exported challenger with the champion under the Phase 3 rules."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from src.api.predictor import Predictor
from src.models.quality_gate import evaluate_quality
from src.monitoring.performance import evaluate_performance

ROOT = Path(__file__).resolve().parents[2]
PROMOTION_RULE = {"recall_must_not_decrease": True, "new_auc_tolerance": 0.01,
                  "old_auc_floor": 0.80, "hard_quality_checks_required": True}


def promotion_decision(champion, challenger, hard_gate_passed):
    """Report eligibility for human review using the team's four conditions."""
    reasons = []
    values = {"champion_new_recall": champion["2021-2023"]["recall"],
              "challenger_new_recall": challenger["2021-2023"]["recall"],
              "champion_new_auc": champion["2021-2023"]["roc_auc"],
              "challenger_new_auc": challenger["2021-2023"]["roc_auc"],
              "challenger_old_auc": challenger["2017-2018"]["roc_auc"],
              "hard_quality_checks_passed": bool(hard_gate_passed)}
    metrics = [value for name, value in values.items() if name != "hard_quality_checks_passed"]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not math.isfinite(value) or not 0 <= value <= 1 for value in metrics):
        reasons.append("Comparison metrics are unavailable or invalid")
    else:
        if values["challenger_new_recall"] < values["champion_new_recall"]:
            reasons.append("Recall decreased on the 2021-2023 holdout")
        if values["challenger_new_auc"] < values["champion_new_auc"] - PROMOTION_RULE["new_auc_tolerance"]:
            reasons.append("ROC-AUC decreased beyond tolerance on the 2021-2023 holdout")
        if values["challenger_old_auc"] < PROMOTION_RULE["old_auc_floor"]:
            reasons.append("ROC-AUC below 0.80 on the 2017-2018 holdout")
    if not hard_gate_passed:
        reasons.append("Challenger failed hard quality-gate checks")
    return {"promote": not reasons, "reasons": reasons, "values": values,
            "rule": PROMOTION_RULE.copy(), "date": datetime.now(timezone.utc).isoformat(),
            "review_required": True, "champion_alias_changed": False}


def compare(champion, challenger, train, old_test, new_test, out):
    """Both models see the same two holdouts and the existing quality gate."""
    report = {"champion": {}, "challenger": {}, "date": datetime.now(timezone.utc).isoformat()}
    for name, predictor in (("champion", champion), ("challenger", challenger)):
        for cycle, frame in (("2017-2018", old_test), ("2021-2023", new_test)):
            report[name][cycle] = evaluate_performance(predictor, frame)
    gates = {cycle: evaluate_quality(challenger, train, frame, out / f"calibration_{cycle}.png")
             for cycle, frame in (("2017-2018", old_test), ("2021-2023", new_test))}
    report["quality_gates"] = gates
    hard_pass = all(not any(c["hard"] and c["status"] == "FAIL" for c in g["checks"])
                    for g in gates.values())
    return report, promotion_decision(report["champion"], report["challenger"], hard_pass)


def verify_provenance(metadata, artifact, root=ROOT):
    """Reject stale candidate files or data changed after candidate training."""
    inputs = metadata.get("input_sha256")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("Candidate is missing training-data provenance; retrain it")
    allowed = {"data/processed/train.csv", "data/processed/test.csv", "data/production/2021-2023_clean.csv",
               "data/production/train_2021.csv", "data/production/test_2021.csv"}
    if set(inputs) != allowed:
        raise ValueError("Candidate provenance must identify all training and evaluation inputs")
    for name, expected in inputs.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Data changed after candidate training: {name}")
    if hashlib.sha256(Path(artifact).read_bytes()).hexdigest() != metadata.get("artifact_sha256"):
        raise ValueError("Candidate artifact does not match its metadata")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=ROOT / "artifacts/retraining/challenger.skops")
    parser.add_argument("--metadata", type=Path, default=ROOT / "artifacts/retraining/model_info.json")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/retraining")
    args = parser.parse_args(argv)
    champion = Predictor()
    champion.load()
    challenger = Predictor(model_path=args.candidate, info_path=args.metadata)
    challenger.load()
    verify_provenance(challenger.model_info, args.candidate)
    old_train = pd.read_csv(ROOT / "data/processed/train.csv")
    new_train = pd.read_csv(ROOT / "data/production/train_2021.csv")
    old_test = pd.read_csv(ROOT / "data/processed/test.csv")
    new_test = pd.read_csv(ROOT / "data/production/test_2021.csv")
    train = pd.concat([old_train, new_train], ignore_index=True)
    from src.models.retrain import check_participants
    check_participants(train, old_test, new_test)
    args.out.mkdir(parents=True, exist_ok=True)
    report, decision = compare(champion, challenger, train, old_test, new_test, args.out)
    report["source_run_id"] = challenger.model_info.get("source_run_id")
    report["input_sha256"] = challenger.model_info["input_sha256"]
    report["artifact_sha256"] = challenger.model_info["artifact_sha256"]
    for name, result in (("comparison.json", report), ("promotion_decision.json", decision)):
        with (args.out / name).open("w", encoding="utf-8", newline="\n") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
    for model in ("champion", "challenger"):
        for cycle, metrics in report[model].items():
            print(f"{model} {cycle}: AUC={metrics['roc_auc']:.4f}, recall={metrics['recall']:.4f}")
    print("Eligible for promotion review" if decision["promote"] else "Retain champion: " + "; ".join(decision["reasons"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
