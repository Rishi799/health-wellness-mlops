"""Build the body topic table: clinical and anthropometric measurements.

Reads the raw BMX, BPX/BPXO, BPQ, TCHOL, HDL and GHB files of one NHANES cycle,
applies documented clinical cleaning decisions (handling repeated blood-pressure
readings, physiological sanity checks, 1/2 questionnaire conversion, and missingness
preservation), and outputs one row per participant matching the team data contract.

Usage (from project root):
    python -m src.data.body 2017-2018
    python -m src.data.body 2021-2023

Output:
    data/interim/body_<cycle>.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "nhanes"
INTERIM = ROOT / "data" / "interim"

FILE_SUFFIX = {"2017-2018": "J", "2021-2023": "L"}

# Blood pressure file names differ between cycles:
# 2017-2018: auscultatory / manual cuff (BPX)
# 2021-2023: oscillometric / automated device (BPXO)
BP_FILE_PREFIX = {"2017-2018": "BPX", "2021-2023": "BPXO"}

# Team-agreed topic contract schema for body_<cycle>.csv
CONTRACT = [
    "seqn",
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
    "hba1c",
]

# Raw BMX mapping -> contract columns
BMX_COLUMNS = {
    "SEQN": "seqn",
    "BMXHT": "height_cm",
    "BMXWT": "weight_kg",
    "BMXBMI": "bmi",
    "BMXWAIST": "waist_cm",
}


def read_xpt(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read selected columns from a SAS transport file (.xpt).

    SAS files represent true zeros as near-zero floating point numbers (e.g. 5.4e-79);
    snap any values with absolute value < 1e-10 back to 0.0.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required raw file not found: {path}")
    df = pd.read_sas(path, format="xport")[columns]
    return df.mask(df.abs() < 1e-10, 0.0)


def process_bp(folder: Path, cycle: str) -> pd.DataFrame:
    """Process blood pressure and pulse readings.

    Clinical decisions:
    1. 2017-2018 (Manual Auscultation):
       - Up to 4 consecutive readings: BPXSY1..4 (systolic), BPXDI1..4 (diastolic).
       - Pulse: BPXPLS (60-second resting pulse).
       - Diastolic = 0 mmHg represents an unmeasured Korotkoff 5th sound or cuff
         artifact, not a true physiological zero. Any DBP <= 0 is converted to NaN.
       - SBP <= 0 or SBP <= DBP is invalid and converted to NaN.
       - Aggregate multiple valid readings using the arithmetic mean.
    2. 2021-2023 (Automated Oscillometric Device):
       - 3 readings: BPXOSY1..3, BPXODI1..3.
       - Pulse: BPXOPLS1..3 recorded with each reading.
       - Mean of valid readings computed for SBP, DBP, and pulse.
    """
    suffix = FILE_SUFFIX[cycle]
    prefix = BP_FILE_PREFIX[cycle]
    path = folder / f"{prefix}_{suffix}.xpt"

    if cycle == "2017-2018":
        sbp_cols = [f"BPXSY{i}" for i in range(1, 5)]
        dbp_cols = [f"BPXDI{i}" for i in range(1, 5)]
        raw_cols = ["SEQN", "BPXPLS", *sbp_cols, *dbp_cols]
        df = read_xpt(path, raw_cols)

        # Invalidate non-physiological / error values
        for i in range(1, 5):
            s_col, d_col = f"BPXSY{i}", f"BPXDI{i}"
            # DBP <= 0 is an artifact
            df[d_col] = df[d_col].where(df[d_col] > 0)
            df[s_col] = df[s_col].where(df[s_col] > 0)
            # SBP must be strictly greater than DBP when both exist
            invalid_pair = (df[s_col].notna()) & (df[d_col].notna()) & (df[s_col] <= df[d_col])
            df.loc[invalid_pair, [s_col, d_col]] = np.nan

        df["sbp"] = df[sbp_cols].mean(axis=1).round(1)
        df["dbp"] = df[dbp_cols].mean(axis=1).round(1)
        # Pulse is measured directly via BPXPLS
        df["pulse"] = df["BPXPLS"].where(df["BPXPLS"] > 0).round(1)

    else:  # 2021-2023
        sbp_cols = [f"BPXOSY{i}" for i in range(1, 4)]
        dbp_cols = [f"BPXODI{i}" for i in range(1, 4)]
        pls_cols = [f"BPXOPLS{i}" for i in range(1, 4)]
        raw_cols = ["SEQN", *sbp_cols, *dbp_cols, *pls_cols]
        df = read_xpt(path, raw_cols)

        for i in range(1, 4):
            s_col, d_col, p_col = f"BPXOSY{i}", f"BPXODI{i}", f"BPXOPLS{i}"
            df[d_col] = df[d_col].where(df[d_col] > 0)
            df[s_col] = df[s_col].where(df[s_col] > 0)
            df[p_col] = df[p_col].where(df[p_col] > 0)
            invalid_pair = (df[s_col].notna()) & (df[d_col].notna()) & (df[s_col] <= df[d_col])
            df.loc[invalid_pair, [s_col, d_col]] = np.nan

        df["sbp"] = df[sbp_cols].mean(axis=1).round(1)
        df["dbp"] = df[dbp_cols].mean(axis=1).round(1)
        df["pulse"] = df[pls_cols].mean(axis=1).round(1)

    return df[["SEQN", "sbp", "dbp", "pulse"]]


def process_bmx(folder: Path, cycle: str) -> pd.DataFrame:
    """Read and validate body measures (height, weight, BMI, waist circumference)."""
    suffix = FILE_SUFFIX[cycle]
    path = folder / f"BMX_{suffix}.xpt"
    df = read_xpt(path, list(BMX_COLUMNS.keys()))

    # Physiological plausibility: height, weight, BMI, waist must be positive
    for col in ["BMXHT", "BMXWT", "BMXBMI", "BMXWAIST"]:
        df[col] = df[col].where(df[col] > 0)

    return df.rename(columns=BMX_COLUMNS)


def process_bpq(folder: Path, cycle: str) -> pd.DataFrame:
    """Process blood pressure questionnaire (BPQ020: told had high blood pressure).

    Codebook mapping:
      1 (Yes) -> 1
      2 (No)  -> 0
      7 (Refused), 9 (Don't know) -> NaN
    """
    suffix = FILE_SUFFIX[cycle]
    path = folder / f"BPQ_{suffix}.xpt"
    df = read_xpt(path, ["SEQN", "BPQ020"])
    df["told_high_bp"] = df["BPQ020"].map({1: 1, 2: 0}).astype("Int64")
    return df[["SEQN", "told_high_bp"]]


def process_labs(folder: Path, cycle: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process laboratory measurements: Total Cholesterol, HDL, and HbA1c."""
    suffix = FILE_SUFFIX[cycle]
    tchol = read_xpt(folder / f"TCHOL_{suffix}.xpt", ["SEQN", "LBXTC"]).rename(columns={"LBXTC": "total_chol"})
    hdl = read_xpt(folder / f"HDL_{suffix}.xpt", ["SEQN", "LBDHDD"]).rename(columns={"LBDHDD": "hdl"})
    ghb = read_xpt(folder / f"GHB_{suffix}.xpt", ["SEQN", "LBXGH"]).rename(columns={"LBXGH": "hba1c"})

    # Positive values only
    tchol["total_chol"] = tchol["total_chol"].where(tchol["total_chol"] > 0)
    hdl["hdl"] = hdl["hdl"].where(hdl["hdl"] > 0)
    ghb["hba1c"] = ghb["hba1c"].where(ghb["hba1c"] > 0)

    return tchol, hdl, ghb


def build(cycle: str) -> pd.DataFrame:
    """Build the harmonized body topic dataset for a given cycle."""
    folder = RAW / cycle
    if not folder.exists():
        raise FileNotFoundError(f"Raw data folder does not exist: {folder}")

    print(f"[{cycle}] Loading and cleaning body & clinical source files...")
    bmx = process_bmx(folder, cycle)
    bp = process_bp(folder, cycle)
    bpq = process_bpq(folder, cycle)
    tchol, hdl, ghb = process_labs(folder, cycle)

    # Standardize SEQN in bmx to uppercase for merge
    bmx = bmx.rename(columns={"seqn": "SEQN"})

    # Merge on SEQN using outer joins so any participant with clinical data is captured
    body = bmx.merge(bp, on="SEQN", how="outer", validate="one_to_one")
    body = body.merge(bpq, on="SEQN", how="outer", validate="one_to_one")
    body = body.merge(tchol, on="SEQN", how="outer", validate="one_to_one")
    body = body.merge(hdl, on="SEQN", how="outer", validate="one_to_one")
    body = body.merge(ghb, on="SEQN", how="outer", validate="one_to_one")

    # Rename SEQN to standard contract 'seqn'
    body = body.rename(columns={"SEQN": "seqn"})
    body["seqn"] = body["seqn"].astype("int64")
    body = body.sort_values("seqn").reset_index(drop=True)

    # Reorder columns to match the team data contract exactly
    body = body[CONTRACT]

    # Integrity assertions
    assert body["seqn"].is_unique, "seqn must be strictly unique"
    assert list(body.columns) == CONTRACT, f"Columns do not match contract: {list(body.columns)}"

    print(f"[{cycle}] Total participants in body dataset: {len(body):,}")
    print(f"[{cycle}] Summary of missingness per clinical feature:")
    for col in CONTRACT[1:]:
        missing_count = body[col].isna().sum()
        pct = (missing_count / len(body)) * 100
        print(f"  - {col:15s}: {missing_count:5d} missing ({pct:5.1f}%)")

    return body


def main(cycle: str) -> None:
    body = build(cycle)
    INTERIM.mkdir(parents=True, exist_ok=True)
    out = INTERIM / f"body_{cycle}.csv"
    body.to_csv(out, index=False)
    print(f"[{cycle}] Saved clean body table to {out.relative_to(ROOT)} ({len(body):,} rows)")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in FILE_SUFFIX:
        sys.exit(f"Usage: python -m src.data.body <cycle>   (one of: {', '.join(FILE_SUFFIX)})")
    main(sys.argv[1])
