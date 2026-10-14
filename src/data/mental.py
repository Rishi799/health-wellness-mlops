from pathlib import Path

import pandas as pd


PHQ9_COLUMNS = [f"DPQ{i:03d}" for i in range(10, 100, 10)]

SEVERITY_LABELS = [
    (0, 4, "minimal"),
    (5, 9, "mild"),
    (10, 14, "moderate"),
    (15, 19, "moderately severe"),
    (20, 27, "severe"),
]


def process_mental_health(input_path: str, output_path: str) -> pd.DataFrame:
    """Create PHQ-9 scores from NHANES depression questionnaire responses."""
    df = pd.read_sas(input_path)

    required = ["SEQN", *PHQ9_COLUMNS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert tiny floating-point values to zero and responses to numeric.
    items = df[PHQ9_COLUMNS].apply(pd.to_numeric, errors="coerce")
    items = items.mask(items.abs() < 1e-10, 0)

    # NHANES codes 7 (refused) and 9 (don't know) are treated as missing.
    items = items.replace({7: pd.NA, 9: pd.NA})

    # Require all nine answers for a complete PHQ-9 score.
    complete = items.notna().all(axis=1)
    scores = items.sum(axis=1).where(complete)

    result = pd.DataFrame({
        "seqn": df["SEQN"],
        "phq9_score": scores,
    })

    def categorize(score):
        if pd.isna(score):
            return pd.NA
        for low, high, label in SEVERITY_LABELS:
            if low <= score <= high:
                return label
        return pd.NA

    result["phq9_category"] = result["phq9_score"].apply(categorize)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    source = root / "data/raw/nhanes/2017-2018/DPQ_J.xpt"
    destination = root / "data/interim/mental_2017-2018.csv"

    result = process_mental_health(str(source), str(destination))
    print(f"Saved: {destination}")
    print(f"Rows: {len(result)}")
    print(f"Complete PHQ-9 scores: {result['phq9_score'].notna().sum()}")
    print("\nCategory counts:")
    print(result["phq9_category"].value_counts(dropna=False))
