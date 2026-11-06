"""Model export utility for champion model (Phase 2 Task C).

Loads the champion Logistic Regression pipeline from MLflow Model Registry,
exports the complete preprocessing + classifier pipeline to skops,
generates metadata in models/model_info.json, and verifies prediction parity.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import skops.io as sio

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_NAME = "cardiometabolic-risk-model"
DEFAULT_ALIAS = "champion"
DEFAULT_FALLBACK_VERSION = "2"
DEFAULT_OUTPUT_DIR = ROOT / "models"
DEFAULT_TEST_CSV = ROOT / "data/processed/test.csv"
DEFAULT_THRESHOLD = 0.5

# Explicit approved trusted types for skops deserialization.
# Under no circumstances should untrusted types be blindly trusted.
APPROVED_SKOPS_TRUSTED_TYPES: List[str] = [
    "numpy.dtype",
]

EXPECTED_FEATURES: List[str] = [
    "age",
    "sex",
    "education",
    "income_ratio",
    "height_cm",
    "weight_kg",
    "bmi",
    "waist_cm",
    "sbp",
    "dbp",
    "pulse",
    "told_high_bp",
    "total_chol",
    "hdl",
    "smoking_status",
    "alcohol_drinks_week",
    "exercise_min_week",
    "sedentary_min_day",
    "sleep_hours",
    "phq9_score",
    "phq9_category",
]


def load_champion_model(
    client: Optional[MlflowClient] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    alias: str = DEFAULT_ALIAS,
    fallback_version: str = DEFAULT_FALLBACK_VERSION,
) -> Tuple[Any, Dict[str, Any]]:
    """Load champion model from MLflow registry using alias with version fallback.

    Requirement 1: Try loading via models:/<model_name>@<alias>
    Requirement 2: If alias loading fails, use registered version 2 explicitly.
    """
    if client is None:
        client = MlflowClient()

    model = None
    resolved_version = str(fallback_version)
    load_source = None

    # Step 1: Attempt to load using alias
    alias_uri = f"models:/{model_name}@{alias}"
    try:
        model = mlflow.sklearn.load_model(alias_uri)
        load_source = alias_uri
        try:
            mv = client.get_model_version_by_alias(model_name, alias)
            resolved_version = str(mv.version)
        except Exception:
            resolved_version = str(fallback_version)
    except Exception as exc:
        warnings.warn(
            f"Failed to load model using alias URI '{alias_uri}' ({type(exc).__name__}: {exc}). "
            f"Falling back to explicit registered version '{fallback_version}'."
        )

    # Step 2: Fallback to version if alias loading failed
    if model is None:
        version_uri = f"models:/{model_name}/{fallback_version}"
        try:
            model = mlflow.sklearn.load_model(version_uri)
            load_source = version_uri
            resolved_version = str(fallback_version)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model using both alias '{alias_uri}' and fallback '{version_uri}': {exc}"
            ) from exc

    # Retrieve model version metadata and tags
    source_run_id = None
    git_commit = "unknown"
    test_roc_auc = 0.0
    test_recall = 0.0
    threshold = DEFAULT_THRESHOLD

    try:
        mv_info = client.get_model_version(model_name, resolved_version)
        source_run_id = getattr(mv_info, "run_id", None)
        tags = getattr(mv_info, "tags", {}) or {}
        git_commit = tags.get("git_commit", "unknown")
        test_roc_auc = float(tags.get("roc_auc", 0.0))
        test_recall = float(tags.get("recall", 0.0))
        threshold = float(tags.get("threshold", DEFAULT_THRESHOLD))
    except Exception as exc:
        warnings.warn(f"Could not retrieve full ModelVersion metadata for version {resolved_version}: {exc}")

    # Fallback to run metrics if needed
    if source_run_id:
        try:
            run = client.get_run(source_run_id)
            if git_commit == "unknown":
                git_commit = run.data.tags.get("git_commit", "unknown")
            if test_roc_auc == 0.0 and "roc_auc" in run.data.metrics:
                test_roc_auc = float(run.data.metrics["roc_auc"])
            if test_recall == 0.0 and "recall" in run.data.metrics:
                test_recall = float(run.data.metrics["recall"])
        except Exception:
            pass

    details = {
        "model_name": model_name,
        "model_version": resolved_version,
        "load_source": load_source,
        "source_run_id": source_run_id,
        "git_commit": git_commit,
        "test_roc_auc": test_roc_auc,
        "test_recall": test_recall,
        "threshold": threshold,
    }

    return model, details


def get_feature_names(pipeline: Any) -> List[str]:
    """Extract ordered 21 input feature names from pipeline or verified default."""
    if hasattr(pipeline, "feature_names_in_"):
        names = list(pipeline.feature_names_in_)
        if len(names) == len(EXPECTED_FEATURES):
            return names
    if hasattr(pipeline, "named_steps") and "preprocessing" in pipeline.named_steps:
        prep = pipeline.named_steps["preprocessing"]
        if hasattr(prep, "feature_names_in_"):
            names = list(prep.feature_names_in_)
            if len(names) == len(EXPECTED_FEATURES):
                return names
    return list(EXPECTED_FEATURES)


def export_pipeline_skops(pipeline: Any, output_path: Union[str, Path]) -> Path:
    """Export the complete pipeline including preprocessing to a .skops file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(pipeline, output_path)
    return output_path


def load_skops_pipeline(
    skops_path: Union[str, Path],
    trusted_types: Optional[List[str]] = None,
) -> Any:
    """Load exported skops model verifying strictly against approved trusted types.

    Requirement adjustment 1: Inspect and explicitly approve required trusted types.
    Do not blindly trust all untrusted types.
    """
    skops_path = Path(skops_path)
    detected_untrusted = sio.get_untrusted_types(file=skops_path)

    approved = set(trusted_types if trusted_types is not None else APPROVED_SKOPS_TRUSTED_TYPES)
    unapproved = [t for t in detected_untrusted if t not in approved]

    if unapproved:
        raise SecurityError(
            f"Security policy violation: detected unapproved untrusted types in {skops_path}: {unapproved}. "
            f"Only explicitly approved types are permitted: {sorted(approved)}"
        )

    return sio.load(skops_path, trusted=list(approved))


class SecurityError(Exception):
    """Raised when skops artifact contains unapproved types."""
    pass


def create_model_info_dict(
    details: Dict[str, Any],
    features: List[str],
    export_timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Format metadata dictionary according to Task C requirements."""
    if export_timestamp is None:
        export_timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "model_name": details.get("model_name", DEFAULT_MODEL_NAME),
        "model_version": str(details.get("model_version", DEFAULT_FALLBACK_VERSION)),
        "source_run_id": details.get("source_run_id"),
        "git_commit": details.get("git_commit", "unknown"),
        "features": list(features),
        "decision_threshold": float(details.get("threshold", DEFAULT_THRESHOLD)),
        "metrics": {
            "test_roc_auc": round(float(details.get("test_roc_auc", 0.0)), 4),
            "test_recall": round(float(details.get("test_recall", 0.0)), 4),
        },
        "test_roc_auc": round(float(details.get("test_roc_auc", 0.0)), 4),
        "test_recall": round(float(details.get("test_recall", 0.0)), 4),
        "export_date": export_timestamp,
    }


def export_model_info_json(
    info_dict: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """Write model_info.json with 2-space indentation."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(info_dict, f, indent=2)
    return output_path


def verify_prediction_parity(
    original_model: Any,
    exported_model: Any,
    test_df: pd.DataFrame,
    features: List[str],
) -> Dict[str, Any]:
    """Verify prediction and probability parity on complete and partially missing inputs.

    Requirement adjustment 2: Verify prediction parity using both complete and
    partially missing inputs, preserving the exact 21-feature order.
    """
    # Subset to exact 21 features in exact order
    clean_sample = test_df[features].copy()

    # Create synthetic missing input sample: copy and introduce missing values across features
    missing_sample = clean_sample.copy()
    rng = np.random.default_rng(42)
    for col in features:
        # Mask roughly 25% of values in this sample to verify imputer behavior
        mask = rng.random(len(missing_sample)) < 0.25
        missing_sample.loc[mask, col] = np.nan

    evaluation_sets = {
        "complete_inputs": clean_sample,
        "partially_missing_inputs": missing_sample,
    }

    parity_summary = {}

    for split_name, df_eval in evaluation_sets.items():
        orig_preds = original_model.predict(df_eval)
        orig_probs = original_model.predict_proba(df_eval)

        exp_preds = exported_model.predict(df_eval)
        exp_probs = exported_model.predict_proba(df_eval)

        preds_match = bool(np.array_equal(orig_preds, exp_preds))
        max_prob_diff = float(np.max(np.abs(orig_probs - exp_probs)))
        probs_match = bool(np.allclose(orig_probs, exp_probs, atol=1e-7))

        if not (preds_match and probs_match):
            raise AssertionError(
                f"Prediction parity failed on {split_name}! "
                f"preds_match={preds_match}, max_prob_diff={max_prob_diff:.2e}"
            )

        parity_summary[split_name] = {
            "samples_evaluated": len(df_eval),
            "predictions_identical": preds_match,
            "probabilities_allclose": probs_match,
            "max_probability_abs_diff": max_prob_diff,
        }

    return parity_summary


def export_and_verify(
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    alias: str = DEFAULT_ALIAS,
    fallback_version: str = DEFAULT_FALLBACK_VERSION,
    test_csv_path: Union[str, Path] = DEFAULT_TEST_CSV,
    sample_size: int = 50,
) -> Dict[str, Any]:
    """Orchestrate model export, metadata writing, and dual-sample parity verification."""
    output_dir = Path(output_dir)
    skops_path = output_dir / "model.skops"
    info_path = output_dir / "model_info.json"

    # 1. Load champion model from MLflow
    client = MlflowClient()
    registered_model, details = load_champion_model(
        client=client,
        model_name=model_name,
        alias=alias,
        fallback_version=fallback_version,
    )

    # 2. Extract ordered features
    features = get_feature_names(registered_model)

    # 3. Export complete pipeline to .skops
    export_pipeline_skops(registered_model, skops_path)

    # 4. Create and write model_info.json
    model_info = create_model_info_dict(details, features)
    export_model_info_json(model_info, info_path)

    # 5. Local reload with strictly inspected approved trusted types
    exported_model = load_skops_pipeline(
        skops_path=skops_path,
        trusted_types=APPROVED_SKOPS_TRUSTED_TYPES,
    )

    # 6. Parity verification with complete and partially missing inputs
    test_df = pd.read_csv(test_csv_path).head(sample_size)
    parity_results = verify_prediction_parity(
        original_model=registered_model,
        exported_model=exported_model,
        test_df=test_df,
        features=features,
    )

    return {
        "skops_path": str(skops_path),
        "model_info_path": str(info_path),
        "model_version": details["model_version"],
        "source_run_id": details["source_run_id"],
        "git_commit": details["git_commit"],
        "load_source": details["load_source"],
        "features_count": len(features),
        "approved_trusted_types": APPROVED_SKOPS_TRUSTED_TYPES,
        "parity_results": parity_results,
    }


def main():
    """CLI entrypoint for Phase 2 Task C model export."""
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Export champion model to skops and json")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Registered model name")
    parser.add_argument("--alias", default=DEFAULT_ALIAS, help="Target model alias")
    parser.add_argument("--fallback-version", default=DEFAULT_FALLBACK_VERSION, help="Fallback version")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--sample-size", type=int, default=50, help="Test rows for parity check")
    args = parser.parse_args()

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "https://dagshub.com/Rishi799/health-wellness-mlops.mlflow",
    )
    mlflow.set_tracking_uri(tracking_uri)

    print("=== Phase 2 Task C — Model Export ===")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Attempting to load champion model '{args.model_name}'...")

    results = export_and_verify(
        output_dir=args.output_dir,
        model_name=args.model_name,
        alias=args.alias,
        fallback_version=args.fallback_version,
        sample_size=args.sample_size,
    )

    print("\nExport completed successfully:")
    print(f"  Exported skops model: {results['skops_path']}")
    print(f"  Exported metadata:    {results['model_info_path']}")
    print(f"  Model Version:        {results['model_version']}")
    print(f"  Source Run ID:        {results['source_run_id']}")
    print(f"  Git Commit:           {results['git_commit']}")
    print(f"  Approved types:       {results['approved_trusted_types']}")
    print("\nParity verification results:")
    for split, metrics in results["parity_results"].items():
        print(f"  [{split}]:")
        for k, v in metrics.items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()

