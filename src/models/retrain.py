"""Train a separate challenger from original and later-cycle training partitions."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import skops.io as sio

from src.api.predictor import Predictor
from src.models.compare_models import EXCLUDE, TARGET, build_preprocessing, get_models
from src.monitoring.performance import evaluate_performance

ROOT = Path(__file__).resolve().parents[2]


def check_participants(train, *tests):
    """Reject missing or duplicate participant keys and any train/test overlap."""
    all_frames = (train, *tests)
    for frame in all_frames:
        if frame.empty or frame["seqn"].isna().any() or not frame["seqn"].is_unique:
            raise ValueError("Each partition needs nonempty, unique participant keys")
        if frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all():
            raise ValueError("Each partition needs binary, non-null labels")
    for index, left in enumerate(all_frames):
        for right in all_frames[index + 1:]:
            if set(left["seqn"]) & set(right["seqn"]):
                raise ValueError("Participant overlap between training and evaluation partitions")


def prepare_splits(old_train, old_test, production):
    if production["seqn"].isna().any() or not production["seqn"].is_unique:
        raise ValueError("Production participant keys must be unique")
    new_train, new_test = train_test_split(production, test_size=0.2, random_state=42,
                                         stratify=production[TARGET])
    train = pd.concat([old_train, new_train], ignore_index=True)
    check_participants(train, old_test, new_test)
    return train, new_train.reset_index(drop=True), new_test.reset_index(drop=True)


def train_challenger(train, features):
    """Build preprocessing and LR from the existing comparison functions."""
    if len(features) != 21 or len(set(features)) != 21 or set(features) & EXCLUDE:
        raise ValueError("Challenger must retain the 21-feature serving contract")
    categorical = [name for name in features if name in ("sex", "education", "smoking_status", "phq9_category")]
    numeric = [name for name in features if name not in categorical]
    frame = train[features].copy()
    for name in categorical:
        frame[name] = frame[name].astype(object).where(frame[name].notna(), np.nan)
    pipeline = Pipeline([("preprocessing", build_preprocessing(numeric, categorical)),
                         ("classifier", get_models()["Logistic Regression"])])
    pipeline.fit(frame, train[TARGET].astype(int))
    return pipeline


def load_trigger(path, force=False):
    if force:
        return {"retrain_needed": True, "reasons": ["forced experiment"]}
    trigger = json.loads(Path(path).read_text(encoding="utf-8"))
    if type(trigger.get("retrain_needed")) is not bool:
        raise ValueError("Trigger must contain a boolean retrain_needed")
    if not isinstance(trigger.get("reasons"), list) or any(not isinstance(reason, str) for reason in trigger["reasons"]):
        raise ValueError("Trigger reasons must be a list of strings")
    return trigger


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trigger", type=Path, default=ROOT / "reports/retraining/trigger.json")
    parser.add_argument("--force", action="store_true", help="Run an explicitly recorded experiment")
    parser.add_argument("--log-mlflow", action="store_true", help="Log the candidate to the configured tracking server")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/retraining")
    args = parser.parse_args(argv)
    trigger = load_trigger(args.trigger, args.force)
    if not trigger["retrain_needed"]:
        print("Trigger requests no retraining. No candidate was fitted.")
        return 0
    old_train = pd.read_csv(ROOT / "data/processed/train.csv")
    old_test = pd.read_csv(ROOT / "data/processed/test.csv")
    production_path = ROOT / "data/production/2021-2023_clean.csv"
    train, new_train, new_test = prepare_splits(old_train, old_test, pd.read_csv(production_path))
    for name, frame in (("train_2021.csv", new_train), ("test_2021.csv", new_test)):
        frame.to_csv(ROOT / "data/production" / name, index=False, lineterminator="\n")
    champion = Predictor()
    champion.load()
    model = train_challenger(train, champion.features)
    candidate = Predictor()
    candidate.model, candidate.features = model, champion.features
    candidate.model_info = deepcopy(champion.model_info)
    candidate.model_version = "challenger"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    candidate.model_info.update(model_version="challenger", git_commit=commit,
                                export_date=datetime.now(timezone.utc).isoformat())
    report = {"date": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
              "training_rows": len(train), "training_rows_2021": len(new_train), "test_rows_2021": len(new_test),
              "seed": 42, "trigger": trigger, "metrics": {}, "source_run_id": None,
              "production_sha256": hashlib.sha256(production_path.read_bytes()).hexdigest()}
    for cycle, frame in (("2017-2018", old_test), ("2021-2023", new_test)):
        metrics = evaluate_performance(candidate, frame)
        prediction = model.predict_proba(frame[champion.features])[:, 1] >= candidate.model_info["decision_threshold"]
        metrics["f1"] = float(f1_score(frame[TARGET], prediction, zero_division=0))
        report["metrics"][cycle] = metrics
    candidate.model_info["metrics"] = {
        "test_roc_auc": report["metrics"]["2017-2018"]["roc_auc"],
        "test_recall": report["metrics"]["2017-2018"]["recall"]}
    candidate.model_info.update(candidate.model_info["metrics"])
    candidate.model_info["evaluation_cycle"] = "2017-2018"
    args.out.mkdir(parents=True, exist_ok=True)
    sio.dump(model, args.out / "challenger.skops")
    inputs = ["data/processed/train.csv", "data/processed/test.csv", "data/production/2021-2023_clean.csv",
              "data/production/train_2021.csv", "data/production/test_2021.csv"]
    provenance = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in inputs}
    report["input_sha256"] = candidate.model_info["input_sha256"] = provenance
    candidate.model_info["artifact_sha256"] = hashlib.sha256((args.out / "challenger.skops").read_bytes()).hexdigest()
    if args.log_mlflow:
        import mlflow
        import mlflow.sklearn
        load_dotenv(ROOT / ".env")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/Rishi799/health-wellness-mlops.mlflow"))
        mlflow.set_experiment("cardiometabolic-retraining")
        with mlflow.start_run(run_name="Logistic Regression challenger") as run:
            report["source_run_id"] = run.info.run_id
            mlflow.set_tags({"git_commit": commit, "model_name": "Logistic Regression",
                             "data_versions": "data-v1 + NHANES 2021-2023",
                             "production_sha256": report["production_sha256"],
                             "trigger_reasons": json.dumps(trigger["reasons"])})
            mlflow.log_params({"train_rows": len(train), "new_train_rows": len(new_train),
                               "n_features": len(champion.features), "seed": 42,
                               **model["classifier"].get_params()})
            for cycle, metrics in report["metrics"].items():
                mlflow.log_metrics({f"{cycle}_{name}": metrics[name] for name in ("roc_auc", "recall", "precision", "f1", "brier")})
            mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["numpy.dtype"])
            mlflow.log_dict(report, "training.json")
    candidate.model_info["source_run_id"] = report["source_run_id"]
    for path, value in ((args.out / "model_info.json", candidate.model_info),
                        (ROOT / "reports/retraining/training.json", report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps({"training_rows": len(train), "source_run_id": report["source_run_id"],
                      "metrics": report["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
