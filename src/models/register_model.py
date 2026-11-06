"""Register champion model in MLflow Model Registry (Phase 2 Task B).

This script identifies the best Logistic Regression model from experiment
'cardiometabolic-model-comparison', registers it under the contract name
'cardiometabolic-risk-model', assigns the 'champion' alias (with fallback
to version number if aliases are unsupported by the remote server), and attaches
required provenance metadata and tags.
"""

import argparse
import os
from pathlib import Path
import warnings
from typing import Any, Dict, Optional, Tuple
from types import SimpleNamespace

from dotenv import load_dotenv
import mlflow
import pandas as pd
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from mlflow.tracking._model_registry import fluent

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = "cardiometabolic-model-comparison"
DEFAULT_MODEL_NAME = "cardiometabolic-risk-model"
DEFAULT_ALIAS = "champion"
DEFAULT_DATASET_VERSION = "data-v1"
TARGET_MODEL_TYPE = "Logistic Regression"
VERIFIED_RUN_ID = "077b778198d448628bf29fbf8c4fd9e6"


def validate_candidate(run: Run) -> dict:
    """Run existing quality policy before registration or champion alias writes.

    This is a regression gate, not a new unbiased model-selection experiment.
    """
    from src.models.quality_gate import evaluate_quality

    if (run.info.status != "FINISHED" or
            run.data.tags.get("model_name", run.data.tags.get("mlflow.runName")) != TARGET_MODEL_TYPE):
        raise ValueError("Candidate must be a finished Logistic Regression run")
    model = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
    candidate = SimpleNamespace(
        model=model, features=list(model.feature_names_in_), model_version="candidate",
        model_info={"model_name": DEFAULT_MODEL_NAME, "decision_threshold": 0.5},
    )
    report = evaluate_quality(candidate, pd.read_csv(ROOT / "data/processed/train.csv"),
                              pd.read_csv(ROOT / "data/processed/test.csv"),
                              ROOT / "reports/figures/candidate_calibration.png")
    for check in report["checks"]:
        print(f"[{check['status']}] candidate {check['check']}")
    if not report["passed"]:
        raise ValueError("Candidate failed the quality gate; champion was not changed")
    return report


def _patch_fluent_get_logged_models(client: MlflowClient) -> None:
    """Workaround DagsHub MLflow limitation where filter_string on search_logged_models returns empty.

    Patches fluent._get_logged_models_from_run so that if the server search with
    filter_string yields no results, it falls back to listing logged models and
    filtering client-side by model_name and source_run_id.
    """
    orig_get_logged = getattr(fluent, "_get_logged_models_from_run", None)
    if orig_get_logged is None:
        return

    def patched_get_logged_models_from_run(source_run: Run, model_name: str):
        res = orig_get_logged(source_run, model_name)
        if not res:
            try:
                all_models = client.search_logged_models(
                    experiment_ids=[source_run.info.experiment_id]
                )
                res = [
                    m for m in all_models
                    if m.name == model_name and m.source_run_id == source_run.info.run_id
                ]
            except Exception:
                pass
        return res

    fluent._get_logged_models_from_run = patched_get_logged_models_from_run


def find_best_logistic_regression_run(
    client: MlflowClient,
    experiment_name: str = DEFAULT_EXPERIMENT,
    preferred_run_id: Optional[str] = VERIFIED_RUN_ID,
) -> Run:
    """Find the best Logistic Regression run in the experiment.

    If preferred_run_id is provided and valid, returns that run.
    Otherwise searches for runs with tags.model_name = 'Logistic Regression'
    ordered by metrics.roc_auc DESC.
    """
    if preferred_run_id:
        try:
            run = client.get_run(preferred_run_id)
            if run.info.status == "FINISHED":
                return run
        except Exception as exc:
            warnings.warn(f"Could not retrieve preferred run {preferred_run_id}: {exc}")

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.model_name = 'Logistic Regression'",
        order_by=["metrics.roc_auc DESC"],
    )

    if not runs:
        # Fallback search if tag differs (e.g. runName)
        all_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.roc_auc DESC"],
        )
        runs = [
            r for r in all_runs
            if r.data.tags.get("model_name") == TARGET_MODEL_TYPE
            or r.data.tags.get("mlflow.runName") == TARGET_MODEL_TYPE
        ]

    if not runs:
        raise ValueError(
            f"No Logistic Regression runs found in experiment '{experiment_name}'"
        )

    return runs[0]


def register_model_artifact(
    run_id: str,
    model_name: str = DEFAULT_MODEL_NAME,
    client: Optional[MlflowClient] = None,
) -> Any:
    """Register the model artifact to MLflow Model Registry."""
    if client is not None:
        _patch_fluent_get_logged_models(client)

    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    return model_version


def set_model_alias(
    client: MlflowClient,
    model_name: str,
    alias: str,
    version: str,
) -> Tuple[bool, str]:
    """Attempt to assign an alias to a registered model version.

    If DagsHub MLflow does not support aliases, catches the exception
    and returns (False, fallback_message).
    """
    try:
        client.set_registered_model_alias(name=model_name, alias=alias, version=str(version))
        return True, f"Alias '{alias}' successfully set on version {version}."
    except Exception as exc:
        msg = (
            f"DagsHub MLflow does not support model aliases ({type(exc).__name__}: {exc}). "
            f"Fallback: referencing champion by registered version number '{version}'."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        return False, msg


def apply_model_metadata(
    client: MlflowClient,
    model_name: str,
    version: str,
    run: Run,
    threshold: float = 0.5,
) -> Dict[str, str]:
    """Apply tags and description to registered model and model version."""
    git_commit = run.data.tags.get("git_commit", "unknown")
    dataset_version = run.data.tags.get("dataset_version", DEFAULT_DATASET_VERSION)
    roc_auc = run.data.metrics.get("roc_auc", 0.0)
    recall = run.data.metrics.get("recall", 0.0)

    tags = {
        "dataset_version": str(dataset_version),
        "git_commit": str(git_commit),
        "roc_auc": f"{roc_auc:.4f}",
        "recall": f"{recall:.4f}",
        "threshold": str(threshold),
        "source_run_id": str(run.info.run_id),
    }

    description = (
        f"Champion Logistic Regression model for cardiometabolic risk awareness (Phase 2). "
        f"Trained on {dataset_version} with 21 features. "
        f"Test ROC-AUC: {roc_auc:.4f}, Test Recall: {recall:.4f}, Decision Threshold: {threshold}."
    )

    # Set tags on model version
    for k, v in tags.items():
        try:
            client.set_model_version_tag(name=model_name, version=str(version), key=k, value=v)
        except Exception as exc:
            warnings.warn(f"Could not set tag {k} on version {version}: {exc}")

    # Set description on model version
    try:
        client.update_model_version(name=model_name, version=str(version), description=description)
    except Exception as exc:
        warnings.warn(f"Could not set description on version {version}: {exc}")

    # Set description on registered model
    try:
        client.update_registered_model(
            name=model_name,
            description="Phase 2 Cardiometabolic Risk Awareness Model Registry (Person 4).",
        )
    except Exception:
        pass

    return tags


def main() -> None:
    """CLI entrypoint for model registration."""
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Register champion model in MLflow Model Registry")
    parser.add_argument("--run-id", default=VERIFIED_RUN_ID, help="Specific run ID to register")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Registered model name")
    parser.add_argument("--alias", default=DEFAULT_ALIAS, help="Model alias to set (e.g. champion)")
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT),
        help="Experiment name to search if run-id not found",
    )
    args = parser.parse_args()

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "https://dagshub.com/Rishi799/health-wellness-mlops.mlflow",
    )
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    print(f"Connecting to MLflow Tracking URI: {tracking_uri}")
    print(f"Looking for run: {args.run_id} (or best in '{args.experiment_name}')...")

    run = find_best_logistic_regression_run(
        client=client,
        experiment_name=args.experiment_name,
        preferred_run_id=args.run_id,
    )
    run_id = run.info.run_id
    git_commit = run.data.tags.get("git_commit", "unknown")
    roc_auc = run.data.metrics.get("roc_auc", 0.0)
    recall = run.data.metrics.get("recall", 0.0)
    train_roc_auc = run.data.metrics.get("train_roc_auc", 0.0)
    train_recall = run.data.metrics.get("train_recall", 0.0)

    print("\nVerified Run Details:")
    print(f"  Run ID: {run_id}")
    print(f"  Git Commit: {git_commit}")
    print(f"  Test ROC-AUC: {roc_auc:.4f}")
    print(f"  Test Recall: {recall:.4f}")
    print(f"  Train ROC-AUC: {train_roc_auc:.4f}")
    print(f"  Train Recall: {train_recall:.4f}")

    validate_candidate(run)
    try:
        previous_version = str(client.get_model_version_by_alias(args.model_name, args.alias).version)
    except Exception:
        previous_version = None

    print(f"\nRegistering model as '{args.model_name}'...")
    model_version = register_model_artifact(
        run_id=run_id,
        model_name=args.model_name,
        client=client,
    )
    version = str(model_version.version)
    print(f"Registered version: {version}")

    tags = apply_model_metadata(
        client=client,
        model_name=args.model_name,
        version=version,
        run=run,
        threshold=0.5,
    )
    print("Metadata & tags successfully applied:")
    for k, v in tags.items():
        print(f"  {k} = {v}")

    if previous_version:
        client.set_model_version_tag(args.model_name, version, "previous_champion_version", previous_version)
    alias_ok, alias_msg = set_model_alias(client, args.model_name, args.alias, version)
    print(f"Alias status: {alias_msg}")

    print(f"\nModel registration complete: {args.model_name} version {version}")


if __name__ == "__main__":
    main()
