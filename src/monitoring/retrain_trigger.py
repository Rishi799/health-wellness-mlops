"""Combine batch drift and labelled performance into a retraining decision."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RULES = {"share_drifted_features": 0.30, "recall": 0.777, "roc_auc": 0.80}


def decide(drift, performance):
    """The three team rules use OR, with inclusive drift and strict metric floors."""
    values = {"share_drifted_features": drift["share_drifted_features"],
              "recall": performance["recall"], "roc_auc": performance["roc_auc"]}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be a finite number in [0, 1]")
    # Metadata is optional for minimal contract files, but must agree when supplied.
    for key in ("data_sha256", "model_version", "batch"):
        if key in drift and key in performance and drift[key] != performance[key]:
            raise ValueError(f"Drift and performance refer to different {key}")
    reasons = []
    if values["share_drifted_features"] >= RULES["share_drifted_features"]:
        reasons.append("data drift")
    if values["recall"] < RULES["recall"]:
        reasons.append("recall degradation")
    if values["roc_auc"] < RULES["roc_auc"]:
        reasons.append("ROC-AUC below quality-gate floor")
    return {"retrain_needed": bool(reasons), "reasons": reasons, "values": values,
            "rules": RULES.copy(), "date": datetime.now(timezone.utc).isoformat(),
            "batch": drift.get("batch"), "model_version": performance.get("model_version"),
            "data_sha256": performance.get("data_sha256")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drift", type=Path, default=ROOT / "reports/drift/drift_summary.json")
    parser.add_argument("--performance", type=Path, default=ROOT / "reports/drift/performance.json")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/retraining/trigger.json")
    args = parser.parse_args(argv)
    try:
        decision = decide(json.loads(args.drift.read_text(encoding="utf-8")),
                          json.loads(args.performance.read_text(encoding="utf-8")))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(decision, output, indent=2, allow_nan=False)
            output.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # A failed run must not leave an old affirmative decision at this path.
        args.out.unlink(missing_ok=True)
        print(f"Cannot evaluate retraining trigger ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    print("Retraining needed: " + ", ".join(decision["reasons"]) if decision["retrain_needed"]
          else "No retraining needed under the agreed rules")
    print(f"Decision saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
