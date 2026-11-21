"""Evaluate the exported champion on labelled production data without refitting it."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.api.predictor import Predictor
from src.models.quality_gate import AGE_GROUPS, TARGET, _metrics

ROOT = Path(__file__).resolve().parents[2]


def evaluate_performance(predictor, frame):
    """Reuse the quality gate's threshold and subgroup metric definitions."""
    if frame.empty or frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all():
        raise ValueError("Evaluation requires nonempty data with binary, non-null labels")
    threshold = float(predictor.model_info["decision_threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid model decision threshold")
    probability_matrix = np.asarray(predictor.model.predict_proba(frame[predictor.features]))
    if (probability_matrix.shape != (len(frame), 2) or
            list(predictor.model.classes_) != [0, 1] or
            not np.isfinite(probability_matrix).all() or
            not ((probability_matrix >= 0) & (probability_matrix <= 1)).all() or
            not np.allclose(probability_matrix.sum(axis=1), 1)):
        raise ValueError("Expected finite n x 2 probabilities for classes [0, 1]")
    probability = probability_matrix[:, 1]
    labels = frame[TARGET].to_numpy(dtype=int)
    result = _metrics(labels, probability, threshold)
    result["brier"] = float(brier_score_loss(labels, probability))
    masks = [("sex", str(sex), frame["sex"].eq(sex).to_numpy()) for sex in (1, 2)]
    masks.extend(("age", name, frame["age"].between(low, high).to_numpy())
                 for name, low, high in AGE_GROUPS)
    result["subgroups"] = [
        {"field": field, "group": group, **_metrics(labels[mask], probability[mask], threshold)}
        for field, group, mask in masks
    ]
    result.update(model_version=predictor.model_version, decision_threshold=threshold,
                  date=datetime.now(timezone.utc).isoformat())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, default=ROOT / "data/production/2021-2023_clean.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/drift/performance.json")
    args = parser.parse_args(argv)
    args.out.unlink(missing_ok=True)
    predictor = Predictor()
    predictor.load()
    report = evaluate_performance(predictor, pd.read_csv(args.current))
    report.update(batch=args.current.name, data_sha256=hashlib.sha256(args.current.read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
