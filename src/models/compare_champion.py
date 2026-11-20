"""Compare an exported challenger with the champion under the Phase 3 rules."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
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


def register_challenger(metadata, decision, client=None):
    """Opt-in registration of an eligible run; only the challenger alias is written."""
    if decision.get("promote") is not True:
        raise ValueError("An ineligible candidate cannot be registered")
    run_id = metadata.get("source_run_id")
    if not run_id:
        raise ValueError("Registration requires a tracked source run")
    from dotenv import load_dotenv
    import mlflow
    from mlflow.tracking import MlflowClient
    from src.models.register_model import DEFAULT_MODEL_NAME, register_model_artifact

    load_dotenv(ROOT / ".env")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/Rishi799/health-wellness-mlops.mlflow"))
    client = client or MlflowClient()
    run = client.get_run(run_id)
    if (run.info.status != "FINISHED" or run.data.tags.get("git_commit") != metadata.get("git_commit") or
            run.data.tags.get("production_sha256") != metadata["input_sha256"]["data/production/2021-2023_clean.csv"]):
        raise ValueError("Tracked run must be finished and match candidate provenance")
    experiment = client.get_experiment(run.info.experiment_id)
    if experiment.name != "cardiometabolic-retraining":
        raise ValueError("Candidate must come from the retraining experiment")
    # Repeated registration commands must reuse the same run's version.
    existing = [version for version in client.search_model_versions(f"name='{DEFAULT_MODEL_NAME}'")
                if version.run_id == run_id]
    version = existing[0] if existing else register_model_artifact(run_id, DEFAULT_MODEL_NAME, client)
    client.set_registered_model_alias(name=DEFAULT_MODEL_NAME, alias="challenger", version=str(version.version))
    client.set_model_version_tag(name=DEFAULT_MODEL_NAME, version=str(version.version), key="review_required", value="true")
    return {"model_name": DEFAULT_MODEL_NAME, "version": str(version.version), "alias": "challenger",
            "source_run_id": run_id, "champion_alias_changed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=ROOT / "artifacts/retraining/challenger.skops")
    parser.add_argument("--metadata", type=Path, default=ROOT / "artifacts/retraining/model_info.json")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/retraining")
    parser.add_argument("--register", action="store_true", help="Register an eligible tracked run with alias challenger")
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
    if args.register and decision["promote"]:
        decision["registration"] = register_challenger(challenger.model_info, decision)
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
