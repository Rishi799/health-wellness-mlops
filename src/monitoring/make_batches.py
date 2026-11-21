"""Create six seeded production partitions and a labelled synthetic stress batch."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.api.schemas import PredictionInput

ROOT = Path(__file__).resolve().parents[2]


def make_batches(frame, seed=42, count=6):
    """Partition every real row once; stress rows are a separate synthetic sample."""
    if count < 1 or len(frame) < count:
        raise ValueError("Need at least one real row per requested batch")
    if frame["seqn"].isna().any() or not frame["seqn"].is_unique:
        raise ValueError("Participant keys must be present and unique")
    shuffled = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    batches = [shuffled.iloc[index].copy() for index in np.array_split(np.arange(len(shuffled)), count)]
    older = frame.loc[frame["age"].ge(60)]
    if older.empty:
        raise ValueError("Need adults aged at least 60 for the synthetic stress batch")
    shifted = older.sample(n=min(len(older), len(batches[0])), random_state=seed).reset_index(drop=True).copy()
    schema = PredictionInput.model_json_schema()["properties"]
    for name, delta in (("bmi", 3.0), ("sbp", 15.0)):
        bounds = next(item for item in schema[name]["anyOf"] if item.get("type") != "null")
        shifted[name] = (shifted[name] + delta).clip(bounds["minimum"], bounds["maximum"])
    return batches, shifted


def write_batches(source, output, seed=42, count=6):
    frame = pd.read_csv(source)
    batches, shifted = make_batches(frame, seed, count)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    expected = {f"batch_{index:02d}.csv" for index in range(1, count + 1)} | {"batch_shifted.csv", "manifest.json"}
    # Avoid accidentally mixing an earlier partition count with the current run.
    if any(path.name not in expected for path in output.iterdir()):
        raise ValueError("Batch output directory contains unexpected files; use a clean directory")
    for index, batch in enumerate(batches, start=1):
        batch.to_csv(output / f"batch_{index:02d}.csv", index=False, lineterminator="\n")
    shifted.to_csv(output / "batch_shifted.csv", index=False, lineterminator="\n")
    manifest = {"source": Path(source).name, "seed": seed, "rows": len(frame),
                "batch_rows": [len(batch) for batch in batches],
                "batches_are_temporal": False,
                "synthetic": {"file": "batch_shifted.csv", "rows": len(shifted),
                              "minimum_age": 60, "bmi_offset": 3.0, "sbp_offset": 15.0,
                              "clipping": "API bounds", "labels": "Retained source labels; not counterfactual outcomes"}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/production/2021-2023_clean.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data/production/batches")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args(argv)
    print(json.dumps(write_batches(args.source, args.out, args.seed, args.count), indent=2))


if __name__ == "__main__":
    main()
