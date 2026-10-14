"""Hand-checkable tests for src/data/lifestyle.py (run: pytest tests/)."""

import numpy as np
import pandas as pd
import pytest

from src.data import lifestyle as L


def frame(**cols):
    n = len(next(iter(cols.values())))
    return pd.DataFrame(cols, index=pd.Index(range(1, n + 1), name="SEQN"))


def test_smoking_rule_every_combination():
    smq = frame(
        SMQ020=[2, 1, 1, 1, 1, 7, np.nan, 1],
        SMQ040=[np.nan, 1, 2, 3, 9, np.nan, np.nan, np.nan],
    )
    got = L.clean_smoking(smq).tolist()
    assert got[:4] == ["never", "current", "current", "former"]
    assert all(pd.isna(v) for v in got[4:])


def test_alcohol_conversion_by_hand():
    # code 3 = 3-4 times a week = 182 days/year; 2 drinks per day
    # 182 * 2 / 52 = 7 drinks per week
    alq = frame(
        ALQ111=[1, 2, 1, 1, 1, 1],
        ALQ121=[3, np.nan, 0, 3, 77, 5],
        ALQ130=[2, np.nan, np.nan, 999, 3, 1],
    )
    got = L.clean_alcohol(alq)
    assert got.iloc[0] == pytest.approx(7.0)
    assert got.iloc[1] == 0  # never drank
    assert got.iloc[2] == 0  # no drinking last year
    assert np.isnan(got.iloc[3])  # ALQ130 = 999 is a code
    assert np.isnan(got.iloc[4])  # ALQ121 = 77 is a code
    assert got.iloc[5] == pytest.approx(1.0)  # once a week, 1 drink


def test_activity_2017_skip_pattern_double_and_cap():
    paq = frame(
        PAQ650=[1, 2, 1, 1, 7, 1],
        PAQ655=[3, np.nan, 7, 2, np.nan, 2],
        PAD660=[30, np.nan, 600, 9999, np.nan, 20],
        PAQ665=[2, 1, 2, 2, 2, 1],
        PAQ670=[np.nan, 4, np.nan, np.nan, np.nan, 5],
        PAD675=[np.nan, 15, np.nan, np.nan, np.nan, 30],
    )
    got = L.clean_activity(paq, "2017-2018")
    assert got.iloc[0] == 3 * 30 * 2  # vigorous counts double, moderate gate 'no' -> 0
    assert got.iloc[1] == 4 * 15  # vigorous gate 'no' -> 0
    assert got.iloc[2] == L.CAP_MIN_WEEK  # 7 * 600 * 2 is capped
    assert np.isnan(got.iloc[3])  # 9999 minutes is a code
    assert np.isnan(got.iloc[4])  # gate refused
    assert got.iloc[5] == 2 * 20 * 2 + 5 * 30


def test_activity_2021_units_and_bytes():
    paq = frame(
        PAD790Q=[3, 0, 4, 2, np.nan],
        PAD790U=[b"W", b"", "b'M'", b"D", b""],
        PAD800=[30, np.nan, 30, 10, np.nan],
        PAD810Q=[1, 0, 0, 0, np.nan],
        PAD810U=[b"D", b"", b"", b"", b""],
        PAD820=[20, np.nan, np.nan, np.nan, np.nan],
    )
    got = L.clean_activity(paq, "2021-2023")
    assert got.iloc[0] == 3 * 1 * 30 + 2 * (1 * 7 * 20)  # 90 + 280
    assert got.iloc[1] == 0  # count 0, empty unit -> 0 minutes
    assert got.iloc[2] == pytest.approx(4 * 12 / 52 * 30)
    assert got.iloc[3] == 2 * 7 * 10
    assert np.isnan(got.iloc[4])


def test_decode_unit_variants():
    s = pd.Series([b"W", "b'W'", "w", "", b"", None, np.nan], dtype=object)
    got = L.decode_unit(s).tolist()
    assert got[:3] == ["W", "W", "W"]
    assert all(pd.isna(v) for v in got[3:])


def test_sleep_weighted_and_fallback():
    slq = frame(SLD012=[7.0, 8.0, np.nan, np.nan], SLD013=[9.0, np.nan, 6.0, np.nan])
    got = L.clean_sleep(slq)
    assert got.iloc[0] == pytest.approx((5 * 7 + 2 * 9) / 7, abs=0.01)
    assert got.iloc[1] == 8.0 and got.iloc[2] == 6.0
    assert np.isnan(got.iloc[3])


def test_diet_codes_and_missing_column():
    assert L.clean_diet(frame(DBQ700=[1, 5, 7, 9])).isna().tolist() == [False, False, True, True]
    assert L.clean_diet(frame(OTHER=[1, 2])).isna().all()  # 2021-2023


def test_sedentary_codes():
    got = L.clean_sedentary(frame(PAD680=[480, 9999, 7777, 1320]))
    assert got.isna().tolist() == [False, True, True, False]


def test_fake_tiny_zero_fixed(tmp_path):
    folder = tmp_path / "2017-2018"
    folder.mkdir()
    pd.DataFrame({"SEQN": [1, 2], "PAD680": [5.4e-79, 300.0]}).to_csv(folder / "PAQ_J.csv", index=False)
    df = L.load_raw(tmp_path, "2017-2018", "PAQ")
    assert df.loc[1, "PAD680"] == 0.0


def test_end_to_end_both_cycles(tmp_path):
    from tests.synthetic import make_raw_lifestyle

    raw = tmp_path / "raw"
    for cycle in L.CYCLE_SUFFIX:
        make_raw_lifestyle(raw, cycle, n=300)
        out = L.build_lifestyle(cycle, raw)
        assert list(out.columns) == L.CONTRACT_COLUMNS
        assert out["seqn"].is_unique
        num = out.drop(columns=["seqn", "smoking_status"])
        assert not num.isin([7777, 9999, 999, 777, 99999, 77777]).any().any()
        assert (num.fillna(1).abs().where(num.fillna(1) != 0, 1) >= 1e-6).all().all()
        assert set(out["smoking_status"].dropna()) <= {"never", "former", "current"}
        if cycle == "2021-2023":
            assert out["diet_quality"].isna().all()
