"""Fairlearn subgroup assessment with counts and Wilson recall intervals."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from fairlearn.metrics import MetricFrame, count, selection_rate
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from statsmodels.stats.proportion import proportion_confint

from src.api.predictor import Predictor
from src.monitoring.performance import evaluate_performance

ROOT = Path(__file__).resolve().parents[2]


def subgroup_metrics(labels, probabilities, sensitive, threshold=0.5):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if (not len(labels) or len(labels) != len(probabilities) or len(labels) != len(sensitive) or
            not np.isin(labels, [0, 1]).all() or not np.isfinite(probabilities).all() or
            not ((probabilities >= 0) & (probabilities <= 1)).all()):
        raise ValueError("Fairness assessment requires aligned binary labels and finite probabilities")
    metrics = {"n": count, "positives": lambda y, p: int(np.sum(y)),
               "detected": lambda y, p: int(np.sum((np.asarray(y) == 1) & np.asarray(p))),
               "recall": lambda y, p: np.sum((np.asarray(y) == 1) & np.asarray(p)) / np.sum(y) if np.sum(y) else np.nan,
               "precision": lambda y, p: precision_score(y, p, zero_division=0),
               "selection_rate": selection_rate}
    assessment = MetricFrame(metrics=metrics, y_true=labels, y_pred=probabilities >= threshold,
                             sensitive_features=pd.Series(sensitive, name="group").reset_index(drop=True))
    groups = []
    for name, row in assessment.by_group.iterrows():
        n, positives, detected = (int(row[field]) for field in ("n", "positives", "detected"))
        interval = list(map(float, proportion_confint(detected, positives, method="wilson"))) if positives else None
        groups.append({"group": str(name), "n": n, "positives": positives, "detected": detected,
                       "recall": float(row["recall"]) if positives else None,
                       "recall_ci95_wilson": interval, "precision": float(row["precision"]),
                       "selection_rate": float(row["selection_rate"])})
    observed = [group["recall"] for group in groups if group["recall"] is not None]
    return {"groups": groups, "recall_difference": max(observed) - min(observed) if len(observed) >= 2 else None}


def assess(predictor, frame):
    overall = evaluate_performance(predictor, frame)
    probabilities = predictor.model.predict_proba(frame[predictor.features])[:, 1]
    age_groups = pd.cut(frame["age"], [19, 39, 59, 79, 80], labels=["20-39", "40-59", "60-79", "80"])
    if age_groups.isna().any() or not frame["sex"].isin([1, 2]).all():
        raise ValueError("Subgroup fields must follow the adult serving contract")
    result = {"model_version": predictor.model_version, "n": len(frame), "date": datetime.now(timezone.utc).isoformat(),
              "threshold": predictor.model_info["decision_threshold"], "overall": overall,
              "interval_note": "95% Wilson intervals quantify binomial recall uncertainty within each group"}
    for field, values in (("sex", frame["sex"].astype(int).astype(str)), ("age", age_groups.astype(str))):
        result[field] = subgroup_metrics(frame["cardiometabolic"], probabilities, values, result["threshold"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, default=ROOT / "data/production/2021-2023_clean.csv")
    parser.add_argument("--challenger", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/retraining/fairness.json")
    args = parser.parse_args(argv)
    predictor = (Predictor(ROOT / "artifacts/retraining/challenger.skops", ROOT / "artifacts/retraining/model_info.json")
                 if args.challenger else Predictor())
    predictor.load()
    result = assess(predictor, pd.read_csv(args.current))
    result["dataset"] = args.current.name
    result["data_sha256"] = hashlib.sha256(args.current.read_bytes()).hexdigest()
    result["model_sha256"] = hashlib.sha256(predictor.model_path.read_bytes()).hexdigest()
    result["source_run_id"] = predictor.model_info.get("source_run_id")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"Fairlearn assessment saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
