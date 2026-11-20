"""Run drift, labelled performance, and the retraining decision for one CSV batch."""

import argparse
import hashlib
import json
from pathlib import Path

from src.api.predictor import Predictor
from src.monitoring.drift import evaluate_drift, read_inputs
from src.monitoring.performance import evaluate_performance
from src.monitoring.retrain_trigger import decide

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "data/processed/train.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/drift")
    args = parser.parse_args(argv)
    # A failed attempt must never leave a previous affirmative decision here.
    for name in ("trigger.json", "drift_summary.json", "performance.json", f"{args.current.stem}_drift.html"):
        (args.out / name).unlink(missing_ok=True)
    model = Predictor()
    model.load()
    current = read_inputs(args.current, model)
    if "cardiometabolic" not in current:
        parser.error("This workflow requires labelled CSV data; use src.monitoring.drift for unlabelled JSONL logs")
    snapshot, drift = evaluate_drift(model, read_inputs(args.reference, model), current)
    metadata = {"batch": args.current.name, "data_sha256": hashlib.sha256(args.current.read_bytes()).hexdigest()}
    drift.update(metadata, reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest())
    performance = evaluate_performance(model, current)
    performance.update(metadata)
    decision = decide(drift, performance)
    args.out.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(args.out / f"{args.current.stem}_drift.html"))
    for name, result in (("drift_summary.json", drift), ("performance.json", performance), ("trigger.json", decision)):
        with (args.out / name).open("w", encoding="utf-8", newline="\n") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps({"batch": args.current.name, "drifted_features": len(drift["drifted_features"]),
                      "roc_auc": performance["roc_auc"], "recall": performance["recall"],
                      "retrain_needed": decision["retrain_needed"], "reasons": decision["reasons"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
