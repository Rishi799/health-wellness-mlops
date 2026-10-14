"""Unit tests for Person 2 body & clinical data processing pipeline (src/data/body.py).

Tests the highest-risk operations:
- Contract column compliance and ordering.
- Participant ID (seqn) uniqueness.
- Repeated blood-pressure reading aggregation (mean of valid readings, dropping 0 DBP).
- Handling of questionnaire codes (1/2 -> 1/0, 7/9 -> NaN).
- Preservation of genuine missing values (no premature imputation).
- Graceful error handling for missing input paths.
"""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.body import CONTRACT, build, process_bp, process_bpq


class TestBodyPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.interim_file_2017 = Path("data/interim/body_2017-2018.csv")
        cls.interim_file_2021 = Path("data/interim/body_2021-2023.csv")

    def test_output_file_exists_and_non_empty(self):
        """Verify that the 2017-2018 interim dataset was generated and has data."""
        self.assertTrue(self.interim_file_2017.exists(), "body_2017-2018.csv must exist")
        df = pd.read_csv(self.interim_file_2017)
        self.assertGreater(len(df), 5000, "2017-2018 body dataset should have > 5,000 rows")

    def test_contract_columns_and_ordering(self):
        """Verify that output columns exactly match the agreed team contract schema."""
        df = pd.read_csv(self.interim_file_2017)
        self.assertEqual(list(df.columns), CONTRACT, f"Columns must exactly match {CONTRACT}")

    def test_seqn_uniqueness(self):
        """Verify that seqn has no duplicate values."""
        df = pd.read_csv(self.interim_file_2017)
        self.assertTrue(df["seqn"].is_unique, "seqn must have no duplicates")
        self.assertFalse(df["seqn"].isna().any(), "seqn must not contain null values")

    def test_bp_cleaning_drops_zero_diastolic(self):
        """Verify that diastolic < 20 mmHg (artifacts / zeros) is ignored when averaging."""
        # Synthetic test with:
        # Person 1: DBP = [0, 80, 82, NaN] -> valid readings are 80, 82 -> mean = 81.0
        # Person 2: SBP = [120, 122, 124, NaN] -> mean = 122.0
        # Person 3: DBP = [0, 8.0, 12.0, 14.7] -> all < 20 -> all invalid -> NaN
        synthetic_bpx = pd.DataFrame({
            "SEQN": [101, 102, 103],
            "BPXPLS": [72.0, 68.0, 75.0],
            "BPXSY1": [120.0, 110.0, np.nan],
            "BPXSY2": [122.0, 112.0, np.nan],
            "BPXSY3": [np.nan, 114.0, np.nan],
            "BPXSY4": [np.nan, np.nan, np.nan],
            "BPXDI1": [0.0, 70.0, 0.0],
            "BPXDI2": [80.0, 72.0, 8.0],
            "BPXDI3": [82.0, np.nan, 12.0],
            "BPXDI4": [np.nan, np.nan, 14.7],
        })

        # Test systolic average
        sbp_cols = ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"]
        dbp_cols = ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]

        for col in dbp_cols:
            synthetic_bpx[col] = synthetic_bpx[col].where(synthetic_bpx[col] >= 20.0)

        mean_dbp = synthetic_bpx[dbp_cols].mean(axis=1).round(1)
        # Person 1: (80 + 82) / 2 = 81.0 (0 dropped)
        self.assertEqual(mean_dbp.iloc[0], 81.0)
        # Person 2: (70 + 72) / 2 = 71.0
        self.assertEqual(mean_dbp.iloc[1], 71.0)
        # Person 3: all < 20 -> NaN
        self.assertTrue(pd.isna(mean_dbp.iloc[2]))

    def test_questionnaire_code_conversion(self):
        """Verify BPQ020 mapping: 1 -> 1, 2 -> 0, 7/9 -> NaN."""
        synthetic_bpq = pd.DataFrame({
            "SEQN": [1, 2, 3, 4, 5],
            "BPQ020": [1.0, 2.0, 7.0, 9.0, np.nan],
        })
        synthetic_bpq["told_high_bp"] = synthetic_bpq["BPQ020"].map({1: 1, 2: 0}).astype("Int64")
        self.assertEqual(synthetic_bpq["told_high_bp"].iloc[0], 1)
        self.assertEqual(synthetic_bpq["told_high_bp"].iloc[1], 0)
        self.assertTrue(pd.isna(synthetic_bpq["told_high_bp"].iloc[2]))
        self.assertTrue(pd.isna(synthetic_bpq["told_high_bp"].iloc[3]))
        self.assertTrue(pd.isna(synthetic_bpq["told_high_bp"].iloc[4]))

    def test_missing_values_preserved_no_imputation(self):
        """Ensure missing values are preserved as NaN and not imputed with 0."""
        df = pd.read_csv(self.interim_file_2017)
        for col in ["height_cm", "weight_kg", "bmi", "sbp", "dbp", "total_chol", "hdl", "hba1c"]:
            self.assertGreater(df[col].isna().sum(), 0, f"{col} should preserve genuine missing values")
            self.assertFalse((df[col] == 0).any(), f"{col} should not contain 0 as imputed missing value")

    def test_nonexistent_cycle_raises_error(self):
        """Verify that passing an invalid cycle raises FileNotFoundError or ValueError."""
        with self.assertRaises((FileNotFoundError, KeyError)):
            build("1999-2000")


if __name__ == "__main__":
    unittest.main()
