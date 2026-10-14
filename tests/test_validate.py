"""Validation must pass on good data and fail loudly on bad data."""

import json

import numpy as np
import pandas as pd
import pytest

from src.data import lifestyle as L
from src.data import validate as V
from tests.synthetic import make_merged, make_raw_lifestyle


@pytest.fixture(scope="module")
def good(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    make_raw_lifestyle(raw, "2017-2018", n=400)
    lifestyle = L.build_lifestyle("2017-2018", raw)
    return lifestyle, make_merged(lifestyle)


def failed(report):
    return {(c["column"], c["check"]) for c in report["checks"] if c["status"] == "failed"}


def test_good_lifestyle_passes(good):
    lifestyle, _ = good
    report = V.validate_dataframe(lifestyle, "lifestyle", "2017-2018")
    assert report["passed"], V.format_failures(report)


def test_good_merged_passes(good):
    _, merged = good
    report = V.validate_dataframe(merged, "merged", "2017-2018")
    assert report["passed"], V.format_failures(report)


def test_leftover_code_9999_named_with_column_and_value(good):
    _, merged = good
    bad = merged.copy()
    bad.loc[3, "sedentary_min_day"] = 9999
    report = V.validate_dataframe(bad, "merged", "2017-2018")
    assert not report["passed"]
    assert ("sedentary_min_day", "no_leftover_codes") in failed(report)
    text = "\n".join(V.format_failures(report))
    assert "sedentary_min_day" in text and "9999" in text


def test_duplicate_seqn(good):
    _, merged = good
    bad = pd.concat([merged, merged.iloc[[0]]], ignore_index=True)
    report = V.validate_dataframe(bad, "merged", "2017-2018")
    assert ("seqn", "unique") in failed(report)


def test_missing_and_renamed_column(good):
    _, merged = good
    bad = merged.rename(columns={"sleep_hours": "sleep_hrs"})
    report = V.validate_dataframe(bad, "merged", "2017-2018")
    assert ("sleep_hours", "column_present") in failed(report)


def test_bad_category_range_target_and_tiny(good):
    _, merged = good
    bad = merged.copy()
    bad.loc[0, "smoking_status"] = "sometimes"
    bad.loc[1, "sleep_hours"] = 20
    bad.loc[2, "cardiometabolic"] = 2
    bad.loc[3, "alcohol_drinks_week"] = 5.4e-79
    bad.loc[4, "age"] = 15
    keys = failed(V.validate_dataframe(bad, "merged", "2017-2018"))
    assert ("smoking_status", "allowed_categories") in keys
    assert ("sleep_hours", "range[2,14]") in keys
    assert ("cardiometabolic", "binary_0_1") in keys
    assert ("alcohol_drinks_week", "no_tiny_values") in keys
    assert ("age", "range[20,80]") in keys


def test_all_failures_reported_together(good):
    _, merged = good
    bad = merged.copy()
    bad.loc[0, "sedentary_min_day"] = 9999
    bad.loc[1, "seqn"] = bad.loc[2, "seqn"]
    report = V.validate_dataframe(bad, "merged", "2017-2018")
    assert report["n_failed"] >= 2


def test_missingness_limit(good):
    _, merged = good
    bad = merged.copy()
    bad.loc[: int(len(bad) * 0.6), "pulse"] = np.nan
    assert ("pulse", "missing<=40%") in failed(V.validate_dataframe(bad, "merged", "2017-2018"))


def test_diet_quality_all_missing_allowed_in_2021_only(good):
    lifestyle, _ = good
    df = lifestyle.copy()
    df["diet_quality"] = np.nan
    assert V.validate_dataframe(df, "lifestyle", "2021-2023")["passed"]
    assert not V.validate_dataframe(df, "lifestyle", "2017-2018")["passed"]


def test_target_missing_fails_in_merged(good):
    _, merged = good
    bad = merged.copy()
    bad.loc[0, "cardiometabolic"] = np.nan
    assert ("cardiometabolic", "not_null") in failed(V.validate_dataframe(bad, "merged", "2017-2018"))


def test_cli_exit_codes_and_report(good, tmp_path):
    _, merged = good
    ok_path, bad_path = tmp_path / "ok.csv", tmp_path / "bad.csv"
    merged.to_csv(ok_path, index=False)
    broken = merged.copy()
    broken.loc[0, "sedentary_min_day"] = 9999
    broken.to_csv(bad_path, index=False)

    report = tmp_path / "r.json"
    args = ["--cycle", "2017-2018", "--schema", "merged", "--report", str(report)]
    assert V.main([*args, "--path", str(ok_path)]) == 0
    assert json.loads(report.read_text())["passed"] is True
    assert V.main([*args, "--path", str(bad_path)]) == 1
    assert json.loads(report.read_text())["passed"] is False


def test_missing_merged_file_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "INTERIM_DIR", tmp_path)
    assert V.main(["--cycle", "2017-2018", "--report", str(tmp_path / "r.json")]) == 1
