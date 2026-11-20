"""Partition and synthetic stress-batch behavior on small synthetic tables."""

import json

import numpy as np
import pandas as pd
import pytest

from src.monitoring.make_batches import make_batches, write_batches
from src.monitoring.prepare_production import prepare_production


@pytest.fixture
def production_frame():
    return pd.DataFrame({"seqn": range(24), "age": [25, 40, 60, 70] * 6,
                         "bmi": [23.0, 25.0, 28.0, 30.0] * 6,
                         "sbp": [110.0, 120.0, 140.0, 145.0] * 6,
                         "cardiometabolic": [0, 0, 1, 1] * 6})


def test_all_real_rows_partitioned_once(production_frame):
    batches, _ = make_batches(production_frame)
    rebuilt = pd.concat(batches).sort_values("seqn").reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt, production_frame)
    assert max(map(len, batches)) - min(map(len, batches)) <= 1
    assert len(batches) == 6


def test_seed_reproduces_every_batch(production_frame):
    batches, shifted = make_batches(production_frame)
    again, shifted_again = make_batches(production_frame)
    for left, right in zip(batches, again):
        pd.testing.assert_frame_equal(left, right)
    pd.testing.assert_frame_equal(shifted, shifted_again)
    different, _ = make_batches(production_frame, seed=99)
    assert not batches[0].equals(different[0])


def test_shifted_batch_changes_only_stated_features(production_frame):
    _, shifted = make_batches(production_frame)
    assert shifted.columns.tolist() == production_frame.columns.tolist()
    assert shifted["age"].min() >= 60
    assert shifted["age"].mean() > production_frame["age"].mean()
    assert shifted["bmi"].mean() > production_frame["bmi"].mean()
    assert shifted["sbp"].mean() > production_frame["sbp"].mean()
    originals = production_frame.set_index("seqn").loc[shifted["seqn"]]
    np.testing.assert_array_equal(shifted["cardiometabolic"], originals["cardiometabolic"])
    np.testing.assert_array_equal(shifted["bmi"], originals["bmi"] + 3)
    np.testing.assert_array_equal(shifted["sbp"], originals["sbp"] + 15)


def test_shift_clips_bounds_and_preserves_missingness(production_frame):
    production_frame.loc[production_frame.age >= 60, "bmi"] = [79, np.nan] * 6
    production_frame.loc[production_frame.age >= 60, "sbp"] = [259, np.nan] * 6
    _, shifted = make_batches(production_frame, count=1)
    assert shifted.bmi.dropna().eq(80).all()
    assert shifted.sbp.dropna().eq(260).all()
    assert shifted.bmi.isna().sum() == 6


@pytest.mark.parametrize("case", ["empty", "duplicate", "no_older"])
def test_bad_source_rejected(production_frame, case):
    if case == "empty":
        production_frame = production_frame.iloc[:0]
    elif case == "duplicate":
        production_frame.loc[1, "seqn"] = 0
    else:
        production_frame["age"] = 40
    with pytest.raises(ValueError):
        make_batches(production_frame)


def test_disk_outputs_and_synthetic_manifest(production_frame, tmp_path):
    source = tmp_path / "source.csv"
    production_frame.to_csv(source, index=False)
    output = tmp_path / "batches"
    manifest = write_batches(source, output)
    assert len(list(output.glob("*.csv"))) == 7
    assert manifest["synthetic"]["file"] == "batch_shifted.csv"
    assert manifest["batches_are_temporal"] is False
    assert json.loads((output / "manifest.json").read_text())["seed"] == 42
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    write_batches(source, output)
    assert first == {path.name: path.read_bytes() for path in output.iterdir()}


def test_production_copy_requires_passing_validation(tmp_path):
    columns = ["seqn", "cardiometabolic", *[f"feature_{i}" for i in range(29)]]
    frame = pd.DataFrame([[1, 0, *range(29)]], columns=columns)
    source = tmp_path / "merged.csv"
    frame.to_csv(source, index=False)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"cycle": "2021-2023", "passed": False,
                                 "schemas_validated": ["people", "body", "lifestyle", "mental", "merged"]}))
    with pytest.raises(ValueError, match="passing validation"):
        prepare_production(source, report, source, tmp_path / "production.csv")
    content = json.loads(report.read_text())
    content["passed"] = True
    report.write_text(json.dumps(content))
    output = tmp_path / "production.csv"
    assert prepare_production(source, report, source, output)["rows"] == 1
    assert source.read_bytes() == output.read_bytes()
