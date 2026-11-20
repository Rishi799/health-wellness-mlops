"""Explain the LR pipeline in log-odds and aggregate encoded columns by input."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.api.predictor import Predictor

ROOT = Path(__file__).resolve().parents[2]


def explain(predictor, reference, current, max_rows=500):
    if reference.empty or current.empty or max_rows < 1:
        raise ValueError("Explanations need reference and current rows and a positive row limit")
    pipeline = predictor.model
    preprocessing, classifier = pipeline.steps[0][1], pipeline.steps[-1][1]
    background = reference[predictor.features].sample(n=min(200, len(reference)), random_state=42)
    sample = current[predictor.features].sample(n=min(max_rows, len(current)), random_state=42)
    transformed_reference = preprocessing.transform(background)
    transformed_current = preprocessing.transform(sample)
    if hasattr(transformed_reference, "toarray"):
        transformed_reference = transformed_reference.toarray()
        transformed_current = transformed_current.toarray()
    masker = shap.maskers.Independent(transformed_reference, max_samples=len(background))
    explanation = shap.LinearExplainer(classifier, masker)(transformed_current)
    contributions = np.asarray(explanation.values)
    reconstructed = np.asarray(explanation.base_values) + contributions.sum(axis=1)
    expected = classifier.decision_function(transformed_current)
    residual = float(np.max(np.abs(reconstructed - expected)))
    if not np.isfinite(contributions).all() or residual > 1e-7:
        raise ValueError("SHAP contributions did not reconstruct the LR decision function")
    encoded_names = preprocessing.get_feature_names_out()
    grouped = {name: np.zeros(len(sample)) for name in predictor.features}
    for index, encoded in enumerate(encoded_names):
        name = encoded.split("__", 1)[-1]
        feature = next((feature for feature in sorted(predictor.features, key=len, reverse=True)
                        if name == feature or name.startswith(feature + "_")), None)
        if feature is None:
            raise ValueError(f"Cannot map encoded feature {encoded} to the serving contract")
        grouped[feature] += contributions[:, index]
    importance = [{"feature": feature, "mean_absolute_shap": float(np.abs(values).mean()),
                   "mean_shap": float(values.mean())} for feature, values in grouped.items()]
    importance.sort(key=lambda item: item["mean_absolute_shap"], reverse=True)
    return {"model_version": predictor.model_version, "sample_rows": len(sample),
            "background_rows": len(background), "seed": 42, "units": "positive-class log-odds",
            "method": "LinearExplainer with an Independent masker on fitted transformed inputs",
            "additivity_max_error": residual, "features": importance,
            "date": datetime.now(timezone.utc).isoformat()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, help="Background CSV; defaults to the model's training population")
    parser.add_argument("--current", type=Path, default=ROOT / "data/production/2021-2023_clean.csv")
    parser.add_argument("--challenger", action="store_true")
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/explanations")
    args = parser.parse_args(argv)
    predictor = (Predictor(ROOT / "artifacts/retraining/challenger.skops", ROOT / "artifacts/retraining/model_info.json")
                 if args.challenger else Predictor())
    predictor.load()
    if args.reference is not None:
        reference = pd.read_csv(args.reference)
        reference_name = args.reference.name
    else:
        reference = pd.read_csv(ROOT / "data/processed/train.csv")
        reference_name = "2017-2018 training partition"
        if args.challenger:
            reference = pd.concat([reference, pd.read_csv(ROOT / "data/production/train_2021.csv")], ignore_index=True)
            reference_name = "2017-2018 + 2021-2023 training partitions"
    result = explain(predictor, reference, pd.read_csv(args.current), args.rows)
    result["dataset"] = args.current.name
    result["background_population"] = reference_name
    result["data_sha256"] = hashlib.sha256(args.current.read_bytes()).hexdigest()
    result["model_sha256"] = hashlib.sha256(predictor.model_path.read_bytes()).hexdigest()
    result["source_run_id"] = predictor.model_info.get("source_run_id")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "shap_summary.json").open("w", encoding="utf-8", newline="\n") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(8, 6))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    top = result["features"][:12][::-1]
    axes.barh([item["feature"] for item in top], [item["mean_absolute_shap"] for item in top])
    axes.set(xlabel="Mean absolute SHAP contribution (log-odds)", title=f"Model {predictor.model_version}: {len(top)} leading inputs")
    figure.tight_layout()
    figure.savefig(args.out / "shap_importance.png", dpi=150)
    print(f"SHAP summary saved to {args.out}; additivity error={result['additivity_max_error']:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
