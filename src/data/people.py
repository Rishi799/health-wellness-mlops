"""Build the people table: who is in the dataset and what we predict.

Reads the raw DEMO, MCQ and DIQ files of one NHANES cycle, applies the agreed
decisions (see the decision log in notebooks/p1_people.ipynb) and saves one row
per adult with the contract columns.

Usage (from the project root):
    python -m src.data.people 2017-2018
    python -m src.data.people 2021-2023

Output:
    data/interim/people_<cycle>.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "nhanes"
INTERIM = ROOT / "data" / "interim"

FILE_SUFFIX = {"2017-2018": "J", "2021-2023": "L"}
MIN_AGE = 20  # heart-condition questions are only asked from age 20 (decision 2)

# Raw NHANES column -> contract column name
DEMO_COLUMNS = {"SEQN": "seqn", "RIDAGEYR": "age", "RIAGENDR": "sex",
                "DMDEDUC2": "education", "INDFMPIR": "income_ratio"}
CONDITION_COLUMNS = {"MCQ160B": "heart_failure", "MCQ160C": "coronary_heart_disease",
                     "MCQ160D": "angina", "MCQ160E": "heart_attack", "MCQ160F": "stroke"}
DIABETES_COLUMN = {"DIQ010": "diabetes"}

CONTRACT = ["seqn", "age", "sex", "education", "income_ratio", "diabetes", "heart_failure",
            "coronary_heart_disease", "angina", "heart_attack", "stroke", "cardiometabolic"]
TARGET_PARTS = ["diabetes", *CONDITION_COLUMNS.values()]


def read_xpt(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read selected columns of a SAS transport file.

    SAS stores 0 as a tiny float (5.4e-79); snap those back to exactly 0 (decision 1).
    """
    df = pd.read_sas(path, format="xport")[columns]
    return df.mask(df.abs() < 1e-10, 0.0)


def load(cycle: str) -> pd.DataFrame:
    """Load DEMO, MCQ and DIQ and join them on SEQN, keeping everyone in DEMO."""
    suffix = FILE_SUFFIX[cycle]
    folder = RAW / cycle
    demo = read_xpt(folder / f"DEMO_{suffix}.xpt", list(DEMO_COLUMNS))
    mcq = read_xpt(folder / f"MCQ_{suffix}.xpt", ["SEQN", *CONDITION_COLUMNS])
    diq = read_xpt(folder / f"DIQ_{suffix}.xpt", ["SEQN", *DIABETES_COLUMN])

    # validate="one_to_one" stops the script if any person appears twice in a file
    people = (demo.merge(mcq, on="SEQN", how="left", validate="one_to_one")
                  .merge(diq, on="SEQN", how="left", validate="one_to_one"))
    return people.rename(columns={**DEMO_COLUMNS, **CONDITION_COLUMNS, **DIABETES_COLUMN})


def yes_no(answer: pd.Series) -> pd.Series:
    """NHANES 1 = yes, 2 = no  ->  1 / 0. Codes 7 (refused) and 9 (don't know) -> missing (decisions 6, 7)."""
    return answer.map({1: 1, 2: 0}).astype("Int64")


def diabetes(answer: pd.Series) -> pd.Series:
    """Like yes_no, but 3 = borderline counts as yes (decision 8)."""
    return answer.map({1: 1, 3: 1, 2: 0}).astype("Int64")


def build_target(people: pd.DataFrame) -> pd.Series:
    """1 if any of the six conditions is yes; 0 if all six are no; otherwise missing (unknown)."""
    parts = people[TARGET_PARTS]
    any_yes = (parts == 1).fillna(False).any(axis=1)
    # "all no" must mean all six answered AND all six are no; a missing answer is not a "no"
    all_no = parts.notna().all(axis=1) & (parts == 0).fillna(False).all(axis=1)

    target = pd.Series(pd.NA, index=people.index, dtype="Int64")
    target[all_no] = 0
    target[any_yes] = 1
    return target


def build(cycle: str) -> pd.DataFrame:
    people = load(cycle)
    print(f"[{cycle}] loaded: {len(people):,} people")

    # Decision 2: adults only. Decision 3: age 80 means 80+, kept as 80.
    people = people[people["age"] >= MIN_AGE].copy()
    print(f"[{cycle}] adults {MIN_AGE}+: {len(people):,}")

    # Decision 4: education 7 / 9 are codes, not levels. Decision 5: income left as is (missing stays missing).
    people["education"] = people["education"].where(people["education"].between(1, 5)).astype("Int64")

    for column in CONDITION_COLUMNS.values():
        people[column] = yes_no(people[column])
    people["diabetes"] = diabetes(people["diabetes"])

    people["cardiometabolic"] = build_target(people)

    # Decision 9: drop adults whose target is unknown
    unknown = people["cardiometabolic"].isna()
    print(f"[{cycle}] dropped (unknown target): {unknown.sum():,}")
    people = people[~unknown]

    people = people.astype({"seqn": "int64", "age": "int64", "sex": "int64", "cardiometabolic": "int64"})
    people = people[CONTRACT]

    assert people["seqn"].is_unique, "seqn must be unique"
    positives = people["cardiometabolic"].sum()
    print(f"[{cycle}] final: {len(people):,} adults, {positives:,} at risk ({positives / len(people):.1%})")
    return people


def main(cycle: str) -> None:
    people = build(cycle)
    INTERIM.mkdir(parents=True, exist_ok=True)
    out = INTERIM / f"people_{cycle}.csv"
    people.to_csv(out, index=False)
    print(f"[{cycle}] saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in FILE_SUFFIX:
        sys.exit(f"Usage: python -m src.data.people <cycle>   (one of: {', '.join(FILE_SUFFIX)})")
    main(sys.argv[1])
