"""Audit the four committed holdout assessments without data, models or network."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.monitoring import fairness_followup as followup

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence():
    comparison = json.loads((ROOT / "reports/retraining/comparison.json").read_text())
    info = json.loads((ROOT / "models/model_info.json").read_text())
    assessments = {(model, cycle): json.loads(
        (ROOT / f"reports/fairness/{model}_{suffix}.json").read_text())
        for model in ("champion", "challenger") for cycle, suffix in followup.CYCLES.items()}
    return comparison, assessments, info


def test_four_holdouts_have_all_counts_and_young_adult_uncertainty(evidence):
    report = followup.build_followup(*evidence)
    assert len(report["rows"]) == 24
    young = {(r["model"], r["cycle"]): r for r in report["rows"] if r["group"] == "20-39"}
    assert (young[("champion", "2017-2018")]["detected"],
            young[("champion", "2017-2018")]["positives"]) == (2, 17)
    later = young[("challenger", "2021-2023")]
    assert (later["detected"], later["positives"], later["missed"]) == (3, 14, 11)
    assert later["recall_ci95_wilson"] == pytest.approx([0.0757138671, 0.4758923059])
    assert report["decision_threshold"] == 0.5
    assert "no fresh model evaluation" in report["evidence_type"]


@pytest.mark.parametrize("field,value", [
    ("source_run_id", "wrong-run"), ("data_sha256", "wrong-holdout"),
    ("model_sha256", "0" * 64), ("threshold", 0.3),
])
def test_mixed_run_data_model_or_threshold_is_rejected(evidence, field, value):
    comparison, assessments, info = evidence
    assessments[("challenger", "2021-2023")][field] = value
    with pytest.raises(ValueError):
        followup.build_followup(comparison, assessments, info)


@pytest.mark.parametrize("field,value", [
    ("detected", 4), ("detected", -1), ("detected", True),
    ("recall", 0.9), ("recall_ci95_wilson", [0, 1]), ("n", 999),
])
def test_inconsistent_subgroup_numbers_are_rejected(evidence, field, value):
    comparison, assessments, info = evidence
    assessments[("challenger", "2021-2023")]["age"]["groups"][0][field] = value
    with pytest.raises(ValueError):
        followup.build_followup(comparison, assessments, info)


def test_duplicate_or_missing_group_is_rejected(evidence):
    comparison, assessments, info = evidence
    assessment = assessments[("champion", "2017-2018")]
    assessment["sex"]["groups"][1] = deepcopy(assessment["sex"]["groups"][0])
    with pytest.raises(ValueError, match="groups"):
        followup.build_followup(comparison, assessments, info)


def test_interval_for_no_positives_is_unavailable():
    assert followup.wilson_interval(0, 0) is None
    assert followup.wilson_interval(0, 10)[0] == pytest.approx(0)
    assert followup.wilson_interval(10, 10)[1] == pytest.approx(1)


def test_cli_writes_aggregate_report_and_rejects_stale_success(tmp_path):
    names = ["models/model_info.json", "reports/retraining/comparison.json"]
    names.extend(f"reports/fairness/{model}_{suffix}.json"
                 for model in ("champion", "challenger") for suffix in followup.CYCLES.values())
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    out = tmp_path / "fairness.json"
    args = ["--root", str(tmp_path), "--out", str(out)]
    assert followup.main(args) == 0
    report = json.loads(out.read_text())
    assert len(report["source_report_sha256"]) == 6
    assert len(report["rows"]) == 24
    (tmp_path / names[-1]).write_text("{}")
    assert followup.main(args) == 1
    assert not out.exists()
