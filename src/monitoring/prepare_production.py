"""Copy the validated later-cycle table into the production-data contract path."""

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def prepare_production(source, report_path, reference, output):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    expected = {"people", "body", "lifestyle", "mental", "merged"}
    if (report.get("cycle") != "2021-2023" or report.get("passed") is not True or
            set(report.get("schemas_validated", [])) != expected):
        raise ValueError("Production data requires a passing validation report for all five later-cycle tables")
    frame = pd.read_csv(source)
    columns = pd.read_csv(reference, nrows=0).columns.tolist()
    if len(columns) != 31 or frame.columns.tolist() != columns:
        raise ValueError("Production table must have the same ordered 31 columns as the development table")
    if (frame.empty or frame["seqn"].isna().any() or not frame["seqn"].is_unique or
            not frame["cardiometabolic"].isin([0, 1]).all()):
        raise ValueError("Production table must contain unique participants with known binary labels")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    return {"rows": len(frame), "positives": int(frame["cardiometabolic"].sum()),
            "positive_fraction": float(frame["cardiometabolic"].mean())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/interim/merged_2021-2023.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/validation_2021-2023.json")
    parser.add_argument("--reference", type=Path, default=ROOT / "data/interim/merged_2017-2018.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/production/2021-2023_clean.csv")
    args = parser.parse_args(argv)
    print(json.dumps(prepare_production(args.source, args.report, args.reference, args.output), indent=2))


if __name__ == "__main__":
    main()
