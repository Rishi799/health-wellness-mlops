from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "interim"


def merge_cycle(cycle="2017-2018"):
    people = pd.read_csv(DATA / f"people_{cycle}.csv")
    body = pd.read_csv(DATA / f"body_{cycle}.csv")
    mental = pd.read_csv(DATA / f"mental_{cycle}.csv")

    for name, df in [("people", people), ("body", body), ("mental", mental)]:
        if df["seqn"].isna().any() or df["seqn"].duplicated().any():
            raise ValueError(f"{name} has missing or duplicate seqn values")

    merged = people.merge(
        body, on="seqn", how="left", validate="one_to_one"
    ).merge(
        mental, on="seqn", how="left", validate="one_to_one"
    )

    if merged["cardiometabolic"].isna().any():
        raise ValueError("Target contains missing values")

    output = DATA / f"merged_{cycle}.csv"
    merged.to_csv(output, index=False)
    print(f"Saved: {output}")
    print(f"Rows: {len(merged)} | Columns: {len(merged.columns)}")
    print("Missing values by column:")
    print(merged.isna().sum().to_string())


if __name__ == "__main__":
    merge_cycle()
