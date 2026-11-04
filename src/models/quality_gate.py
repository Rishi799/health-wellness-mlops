"""Check the champion against the team's model-quality agreement.

Run from the repository root: python -m src.models.quality_gate
All checks are reported together. Only hard failures cause exit code 1.
Change the rules below only after a team decision, never to make a run pass.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score

from src.api.predictor import Predictor
from src.api.schemas import PredictionInput
from src.models.compare_models import EXCLUDE, TARGET

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = REPO_ROOT / "data/processed/train.csv"
TEST_PATH = REPO_ROOT / "data/processed/test.csv"
REPORT_PATH = REPO_ROOT / "reports/model_quality.json"
PLOT_PATH = REPO_ROOT / "reports/figures/calibration.png"

# The agreement: thresholds, direction, and whether failure blocks the gate.
RULES = {
    "roc_auc": {"limit": 0.80, "direction": "min", "hard": True},
    "recall": {"limit": 0.75, "direction": "min", "hard": True},
    "auc_improvement": {"limit": 0.20, "direction": "min", "hard": True},
    "no_leakage": {"limit": True, "direction": "equal", "hard": True},
    "input_contract": {"limit": True, "direction": "equal", "hard": True},
    "artifact_features": {"limit": True, "direction": "equal", "hard": True},
    "brier_score": {"limit": 0.20, "direction": "max", "hard": False},
    "subgroup_recall": {"limit": 0.60, "direction": "min", "hard": False},
}
AGE_GROUPS = (("20-39", 20, 39), ("40-59", 40, 59), ("60-79", 60, 79), ("80", 80, 80))


def _check(name: str, value, rule_name: str | None = None, detail: str = "") -> dict:
    rule = RULES[rule_name or name]
    limit = rule["limit"]
    ok = value is not None and {
        "min": lambda: value >= limit,
        "max": lambda: value <= limit,
        "equal": lambda: value == limit,
    }[rule["direction"]]()
    return {
        "check": name,
        "value": value,
        "limit": limit,
        "hard": rule["hard"],
        "status": "PASS" if ok else ("FAIL" if rule["hard"] else "WARN"),
        "detail": detail,
    }


def _metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    predicted = probability >= threshold
    positives = int(y.sum())
    return {
        "n": int(len(y)),
        "positives": positives,
        "recall": float(recall_score(y, predicted, zero_division=0)) if positives else None,
        "precision": float(precision_score(y, predicted, zero_division=0)) if len(y) else None,
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
    }


def _save_calibration(y: np.ndarray, probability: np.ndarray, path: Path) -> None:
    # Agg works in Linux CI without a display; never open an interactive window.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    observed, predicted = calibration_curve(y, probability, n_bins=10, strategy="uniform")
    figure = Figure(figsize=(6, 5))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axes.plot(predicted, observed, "o-", label="Champion (10 bins)")
    axes.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction",
             title="Champion calibration on held-out test data", xlim=(0, 1), ylim=(0, 1))
    axes.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)


def evaluate_quality(predictor: Predictor, train: pd.DataFrame, test: pd.DataFrame,
                     plot_path: Path = PLOT_PATH) -> dict:
    """Evaluate without fitting the champion; fit only the dummy on training labels."""
    features = predictor.features
    artifact_features = list(getattr(predictor.model, "feature_names_in_", features))
    checks = [
        _check("no_leakage", not bool((set(features) | set(artifact_features)) & EXCLUDE),
               detail=f"Excluded columns found: {sorted((set(features) | set(artifact_features)) & EXCLUDE)}"),
        _check("input_contract", len(features) == len(set(features)) and
               set(features) == set(PredictionInput.model_fields)),
        _check("artifact_features", artifact_features == features),
    ]
    report = {
        "date": datetime.now(timezone.utc).isoformat(),
        "model_name": predictor.model_info.get("model_name"),
        "model_version": predictor.model_version,
        "thresholds": RULES,
        "decision_threshold": predictor.model_info.get("decision_threshold"),
        "features": features,
        "checks": checks,
        "values": {},
        "subgroups": [],
        "calibration_plot": None,
    }
    try:
        threshold = float(report["decision_threshold"])
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("decision_threshold must be finite and in [0, 1]")
        for name, frame in (("train", train), ("test", test)):
            if frame.empty or frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all():
                raise ValueError(f"{name} must be nonempty with binary, non-null target labels")
        y = test[TARGET].to_numpy(dtype=int)
        probability_matrix = np.asarray(predictor.model.predict_proba(test[features]))
        if (probability_matrix.shape != (len(test), 2) or
                not np.isfinite(probability_matrix).all() or
                not ((probability_matrix >= 0) & (probability_matrix <= 1)).all() or
                not np.allclose(probability_matrix.sum(axis=1), 1)):
            raise ValueError("predict_proba must return finite n x 2 probabilities summing to 1")
        if list(predictor.model.classes_) != [0, 1]:
            raise ValueError("Model classes must be ordered [0, 1]")
        probability = probability_matrix[:, 1]
        dummy = DummyClassifier(strategy="most_frequent").fit(train[features], train[TARGET])
        dummy_auc = float(roc_auc_score(y, dummy.predict_proba(test[features])[:, 1]))
        values = _metrics(y, probability, threshold)
        values.update(dummy_auc=dummy_auc,
                      auc_improvement=values["roc_auc"] - dummy_auc,
                      brier_score=float(brier_score_loss(y, probability)))
        report["values"] = values
        checks.extend(_check(name, values[name])
                      for name in ("roc_auc", "recall", "auc_improvement", "brier_score"))

        masks = [("sex", str(sex), test["sex"].eq(sex).to_numpy()) for sex in (1, 2)]
        masks.extend(("age", label, test["age"].between(lo, hi).to_numpy())
                     for label, lo, hi in AGE_GROUPS)
        for field, group, mask in masks:
            metrics = _metrics(y[mask], probability[mask], threshold)
            report["subgroups"].append({"field": field, "group": group, **metrics})
            checks.append(_check(f"{field}={group} recall", metrics["recall"], "subgroup_recall",
                                 detail=f"n={metrics['n']}, positives={metrics['positives']}"))
        try:
            _save_calibration(y, probability, plot_path)
            report["calibration_plot"] = str(plot_path)
        except (OSError, ValueError) as exc:
            checks.append({"check": "calibration_plot", "status": "FAIL", "hard": True,
                           "value": None, "detail": f"{type(exc).__name__}: {exc}"})
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        checks.append({"check": "evaluation", "status": "FAIL", "hard": True,
                       "value": None, "detail": f"{type(exc).__name__}: {exc}"})
    report["passed"] = not any(c["status"] == "FAIL" for c in checks)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--calibration-plot", type=Path, default=PLOT_PATH)
    args = parser.parse_args(argv)
    predictor = Predictor()
    errors = []
    try:
        predictor.load()
    except Exception as exc:
        errors.append(f"champion: {type(exc).__name__}: {exc}")
    frames = {}
    for name, path in (("train", TRAIN_PATH), ("test", TEST_PATH)):
        try:
            frames[name] = pd.read_csv(path)
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if errors:
        report = {
            "date": datetime.now(timezone.utc).isoformat(), "passed": False,
            "model_name": predictor.model_info.get("model_name"),
            "model_version": predictor.model_version, "thresholds": RULES,
            "decision_threshold": predictor.model_info.get("decision_threshold"),
            "values": {}, "subgroups": [], "calibration_plot": None,
            "checks": [{"check": "load_inputs", "status": "FAIL", "hard": True,
                        "value": None, "detail": error} for error in errors],
        }
    else:
        report = evaluate_quality(predictor, frames["train"], frames["test"], args.calibration_plot)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for check in report["checks"]:
        print(f"[{check['status']}] {check['check']}: {check['value']} {check.get('detail', '')}")
    for group in report["subgroups"]:
        print(f"[subgroup] {group}")
    print(f"[quality_gate] report written to {args.report}")
    print(f"[quality_gate] {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
