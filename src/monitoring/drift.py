"""Compare serving inputs and model scores with a reproducible Evidently report."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset
import numpy as np
import pandas as pd

from src.api.predictor import Predictor
from src.api.schemas import PredictionInput

ROOT = Path(__file__).resolve().parents[2]
CATEGORICAL = {"sex", "education", "told_high_bp", "smoking_status", "phq9_category"}
DISTANCE_THRESHOLD = 0.1


def read_inputs(path, predictor):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record["model_version"]) != predictor.model_version:
                raise ValueError(f"Log line {number} uses a different model version")
            rows.append(PredictionInput.model_validate(record["inputs"]).model_dump())
        frame = pd.DataFrame(rows)
    else:
        frame = pd.read_csv(path)
    if frame.empty or not set(predictor.features) <= set(frame.columns):
        raise ValueError("Drift inputs require nonempty data with all serving features")
    return frame


def scoring_frame(predictor, frame):
    values = frame[predictor.features].copy().replace({None: np.nan})
    probability = np.asarray(predictor.model.predict_proba(values))
    if (probability.shape != (len(values), 2) or list(predictor.model.classes_) != [0, 1] or
            not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all() or
            not np.allclose(probability.sum(axis=1), 1)):
        raise ValueError("Model returned invalid probabilities")
    # Numeric-coded survey categories must be compared as categories.
    for feature in CATEGORICAL & set(predictor.features):
        numeric_code = feature in {"sex", "education", "told_high_bp"}
        values[feature] = values[feature].map(
            lambda value: (str(float(value)) if numeric_code else str(value)) if pd.notna(value) else np.nan)
    values["prediction"] = probability[:, 1]
    return values


def evaluate_drift(predictor, reference, current):
    features = predictor.features
    if len(features) != 21 or len(set(features)) != 21:
        raise ValueError("Drift denominator must be the 21 serving features")
    ref, cur = scoring_frame(predictor, reference), scoring_frame(predictor, current)
    if any(ref[column].dropna().empty or cur[column].dropna().empty for column in features):
        raise ValueError("Cannot estimate distribution drift for an entirely missing feature")
    categorical = [feature for feature in features if feature in CATEGORICAL]
    numeric = [feature for feature in features if feature not in CATEGORICAL] + ["prediction"]
    definition = DataDefinition(numerical_columns=numeric, categorical_columns=categorical)
    report = Report([DataDriftPreset(columns=features, drift_share=0.30,
                                    num_method="wasserstein", num_threshold=DISTANCE_THRESHOLD,
                                    cat_method="jensenshannon", cat_threshold=DISTANCE_THRESHOLD),
                     ValueDrift(column="prediction", method="wasserstein", threshold=DISTANCE_THRESHOLD)])
    snapshot = report.run(current_data=Dataset.from_pandas(cur, data_definition=definition),
                          reference_data=Dataset.from_pandas(ref, data_definition=definition))
    metrics = snapshot.dict()["metrics"]
    distances = {metric["config"]["column"]: float(metric["value"])
                 for metric in metrics if "column" in metric.get("config", {})}
    if set(distances) != set(features) | {"prediction"}:
        raise ValueError("Evidently returned an incomplete drift report")
    if not all(np.isfinite(list(distances.values()))):
        raise ValueError("Evidently returned non-finite drift distances")
    drifted = [feature for feature in features if distances[feature] >= DISTANCE_THRESHOLD]
    fraction = len(drifted) / len(features)
    summary = {"share_drifted_features": fraction, "drifted_features": drifted,
               "dataset_drift": fraction >= 0.30, "prediction_drift": distances["prediction"] >= DISTANCE_THRESHOLD,
               "model_version": predictor.model_version, "feature_count": len(features),
               "reference_rows": len(reference), "current_rows": len(current),
               "methods": {"numeric": "normalized Wasserstein", "categorical": "Jensen-Shannon distance",
                           "column_threshold": DISTANCE_THRESHOLD, "dataset_threshold": 0.30},
               "distances": distances, "missing_fractions": {
                   feature: {"reference": float(ref[feature].isna().mean()), "current": float(cur[feature].isna().mean())}
                   for feature in features}, "date": datetime.now(timezone.utc).isoformat()}
    return snapshot, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "data/processed/train.csv")
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/drift")
    args = parser.parse_args(argv)
    predictor = Predictor()
    predictor.load()
    snapshot, summary = evaluate_drift(predictor, read_inputs(args.reference, predictor), read_inputs(args.current, predictor))
    summary.update(batch=args.current.name, data_sha256=hashlib.sha256(args.current.read_bytes()).hexdigest(),
                   reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest())
    args.out.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(args.out / f"{args.current.stem}_drift.html"))
    with (args.out / "drift_summary.json").open("w", encoding="utf-8", newline="\n") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"{args.current.name}: {len(summary['drifted_features'])}/21 features drifted; prediction drift={summary['prediction_drift']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
