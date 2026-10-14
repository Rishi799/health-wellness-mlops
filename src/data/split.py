"""Split the merged training table into train and test sets.

Stratified on the target so both sets have the same share of at-risk adults, with a
fixed seed so every laptop gets exactly the same split. Settings come from params.yaml.
All columns are kept; choosing features (and dropping leakage columns) happens in the
model code.

Usage (from the project root):
    python -m src.data.split

Input:  data/interim/merged_<cycle>.csv   (cycle from params.yaml, default 2017-2018)
Output: data/processed/train.csv, data/processed/test.csv
"""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
PARAMS_FILE = ROOT / "params.yaml"
TARGET = "cardiometabolic"


def load_params() -> dict:
    params = yaml.safe_load(PARAMS_FILE.read_text())["split"]
    return {"cycle": params["cycle"], "test_size": float(params["test_size"]), "seed": int(params["seed"])}


def split(merged: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(merged, test_size=test_size, random_state=seed,
                                   stratify=merged[TARGET])
    # Same order every run, so the CSV files are byte-identical (DVC hashes stay stable)
    return train.sort_values("seqn"), test.sort_values("seqn")


def main() -> None:
    p = load_params()
    source = INTERIM / f"merged_{p['cycle']}.csv"
    if not source.exists():
        raise FileNotFoundError(f"{source.name} not found - run: python -m src.data.merge {p['cycle']}")
    merged = pd.read_csv(source)

    train, test = split(merged, p["test_size"], p["seed"])
    assert set(train["seqn"]).isdisjoint(test["seqn"]), "a person is in both train and test"
    assert len(train) + len(test) == len(merged), "rows were lost in the split"

    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("test", test)):
        part.to_csv(PROCESSED / f"{name}.csv", index=False)
        print(f"[split] {name:<5}: {len(part):,} rows, {part[TARGET].mean():.1%} at risk "
              f"-> {(PROCESSED / f'{name}.csv').relative_to(ROOT)}")
    print(f"[split] cycle {p['cycle']}, test_size {p['test_size']}, seed {p['seed']}")


if __name__ == "__main__":
    main()
