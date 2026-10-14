"""Merge the four topic tables into one table: one row per adult, all features + target.

Starts from the people table (adults 20+ with a known target) and LEFT-joins the
body, lifestyle and mental tables on seqn. Nobody is added or lost: people who are
missing from a topic table simply get missing values for that topic's columns.

Usage (from the project root, after the four topic scripts have run):
    python -m src.data.merge 2017-2018
    python -m src.data.merge 2021-2023

Input:  data/interim/{people,body,lifestyle,mental}_<cycle>.csv
Output: data/interim/merged_<cycle>.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"

CYCLES = ("2017-2018", "2021-2023")
BASE = "people"                            # decides who is in the dataset
JOINED = ("body", "lifestyle", "mental")   # joined onto the base, in this order
TARGET = "cardiometabolic"

# Which script builds each topic table (shown when a file is missing)
BUILDERS = {
    "people": "python -m src.data.people {cycle}",
    "body": "python -m src.data.body {cycle}",
    "lifestyle": "python -m src.data.lifestyle --cycle {cycle}",
    "mental": "python -m src.data.mental",
}


def load_topic(topic: str, cycle: str) -> pd.DataFrame:
    """Read one topic table and make sure it can be joined safely."""
    path = INTERIM / f"{topic}_{cycle}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found - build it first with: "
                                f"{BUILDERS[topic].format(cycle=cycle)}")
    df = pd.read_csv(path)
    # Some tables store seqn as 93703.0; make it a whole number everywhere so keys match
    df["seqn"] = df["seqn"].astype("int64")
    if not df["seqn"].is_unique:
        raise ValueError(f"{path.name}: seqn is not unique")
    return df


def merge(cycle: str) -> pd.DataFrame:
    merged = load_topic(BASE, cycle)
    n_people = len(merged)
    print(f"[{cycle}] {BASE:<9}: {n_people:,} adults (start)")

    for topic in JOINED:
        table = load_topic(topic, cycle)
        overlap = (set(table.columns) & set(merged.columns)) - {"seqn"}
        if overlap:
            raise ValueError(f"{topic}: columns already present in the merged table: {sorted(overlap)}")

        matched = merged["seqn"].isin(table["seqn"]).sum()
        # validate="one_to_one" stops the script if either side has a duplicate seqn
        merged = merged.merge(table, on="seqn", how="left", validate="one_to_one")
        if len(merged) != n_people:
            raise AssertionError(f"row count changed after joining {topic}: {len(merged):,}")
        print(f"[{cycle}] + {topic:<9}: {table.shape[1] - 1} columns, "
              f"{matched:,} of {n_people:,} adults found ({matched / n_people:.1%})")

    if merged[TARGET].isna().any():
        raise ValueError(f"{TARGET} has missing values - the people table must drop unknown targets")

    positives = int(merged[TARGET].sum())
    print(f"[{cycle}] merged: {len(merged):,} rows x {merged.shape[1]} columns, "
          f"{positives:,} at risk ({positives / len(merged):.1%})")
    return merged


def main(cycle: str) -> None:
    merged = merge(cycle)
    out = INTERIM / f"merged_{cycle}.csv"
    merged.to_csv(out, index=False)
    print(f"[{cycle}] saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in CYCLES:
        sys.exit(f"Usage: python -m src.data.merge <cycle>   (one of: {', '.join(CYCLES)})")
    main(sys.argv[1])
