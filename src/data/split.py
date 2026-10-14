"""Create reproducible, stratified train/test CSV files."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data/interim/merged_2017-2018.csv"
OUTPUT_DIR = ROOT / "data/processed"
TARGET = "cardiometabolic"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Merged dataset not found: {INPUT_PATH}")

    data = pd.read_csv(INPUT_PATH)

    if TARGET not in data.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    if data[TARGET].isna().any():
        raise ValueError("Target contains missing values")
    if data[TARGET].nunique() != 2:
        raise ValueError("Expected a binary classification target")

    train, test = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data[TARGET],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print(f"Training rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print("\nTarget proportions:")
    print("Train:", train[TARGET].value_counts(normalize=True).round(4).to_dict())
    print("Test: ", test[TARGET].value_counts(normalize=True).round(4).to_dict())
    print(f"\nSaved files in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
