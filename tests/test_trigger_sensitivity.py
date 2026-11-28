"""Synthetic counterfactual cases; never mutate the production trigger rules."""

from copy import deepcopy
import json

import pytest

from src.monitoring import trigger_sensitivity as sensitivity
from src.monitoring.retrain_trigger import RULES, decide


def inputs(drifted=0, recall=0.9, auc=0.9):
    drift = {"feature_count": 10, "drifted_features": [f"f{i}" for i in range(drifted)],
             "share_drifted_features": drifted / 10,
             "model_version": "2", "batch": "synthetic.csv", "data_sha256": "same"}
    performance = {"n": 20, "recall": recall, "roc_auc": auc, "model_version": "2",
                   "batch": "synthetic.csv", "data_sha256": "same"}
    return drift, performance


@pytest.mark.parametrize("drifted,recall,auc,reason", [
    (5, 0.9, 0.9, "data drift"),
    (0, 0.776, 0.9, "recall degradation"),
    (0, 0.9, 0.799, "ROC-AUC below quality-gate floor"),
])
def test_each_rule_fires_alone(drifted, recall, auc, reason):
    report = sensitivity.analyse_batch(*inputs(drifted, recall, auc))
    for scenario in report["scenarios"].values():
        assert scenario["retrain_needed"]
        assert scenario["reasons"] == [reason]


@pytest.mark.parametrize("drifted,threshold", [(3, "0.30"), (4, "0.40"), (5, "0.50")])
def test_exact_drift_threshold_is_inclusive(drifted, threshold):
    report = sensitivity.analyse_batch(*inputs(drifted))
    assert report["scenarios"][threshold]["reasons"] == ["data drift"]


@pytest.mark.parametrize("drifted,threshold", [(2, "0.30"), (3, "0.40"), (4, "0.50")])
def test_below_drift_threshold_does_not_trigger(drifted, threshold):
    report = sensitivity.analyse_batch(*inputs(drifted))
    assert not report["scenarios"][threshold]["retrain_needed"]


def test_performance_boundaries_and_negative_control_do_not_trigger():
    report = sensitivity.analyse_batch(*inputs(0, RULES["recall"], RULES["roc_auc"]))
    assert all(not s["retrain_needed"] and not s["reasons"] for s in report["scenarios"].values())


def test_all_reasons_are_retained_and_performance_rules_stay_fixed():
    before = deepcopy(RULES)
    report = sensitivity.analyse_batch(*inputs(4, 0.7, 0.75))
    assert len(report["scenarios"]["0.30"]["reasons"]) == 3
    assert report["scenarios"]["0.50"]["reasons"] == [
        "recall degradation", "ROC-AUC below quality-gate floor"]
    for scenario in report["scenarios"].values():
        assert scenario["rules"]["recall"] == 0.777
        assert scenario["rules"]["roc_auc"] == 0.80
    assert RULES == before == {"share_drifted_features": 0.30, "recall": 0.777, "roc_auc": 0.80}


def test_mixed_batch_evidence_is_rejected():
    drift, performance = inputs()
    performance["data_sha256"] = "another batch"
    with pytest.raises(ValueError, match="different"):
        sensitivity.analyse_batch(drift, performance)


def test_count_share_disagreement_is_rejected():
    drift, performance = inputs(3)
    drift["share_drifted_features"] = 0.4
    with pytest.raises(ValueError, match="count and share"):
        sensitivity.analyse_batch(drift, performance)


def write_reports(directory):
    manifest = []
    for batch in sensitivity.BATCHES:
        drift, performance = inputs(0 if batch == "negative_control" else 4)
        (directory / batch).mkdir(parents=True)
        (directory / batch / "drift_summary.json").write_text(json.dumps(drift))
        (directory / batch / "performance.json").write_text(json.dumps(performance))
        decision = decide(drift, performance)
        manifest.append({"batch": batch, "n": performance["n"], "drifted": len(drift["drifted_features"]),
                         "recall": performance["recall"], "roc_auc": performance["roc_auc"],
                         "retrain_needed": decision["retrain_needed"], "reasons": decision["reasons"]})
    (directory / "batch_results.json").write_text(json.dumps(manifest))


def test_cli_writes_nine_batches_and_readable_causes(tmp_path, capsys):
    directory = tmp_path / "drift"
    write_reports(directory)
    out = tmp_path / "sensitivity.json"
    assert sensitivity.main(["--reports-dir", str(directory), "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert len(report["batches"]) == 9
    assert len(report["source_report_sha256"]) == 19
    control = report["batches"][-1]
    assert all(not s["retrain_needed"] for s in control["scenarios"].values())
    assert report["analysis_only"]
    output = capsys.readouterr().out
    assert "YES:D" in output and "negative_control" in output and "REPORT ONLY" in output


def test_missing_input_has_clear_error_and_removes_stale_output(tmp_path, capsys):
    write_reports(tmp_path)
    (tmp_path / "batch_03/performance.json").unlink()
    out = tmp_path / "sensitivity.json"
    out.write_text('{"analysis_only": true}')
    assert sensitivity.main(["--reports-dir", str(tmp_path), "--out", str(out)]) == 2
    error = capsys.readouterr().err
    assert "FileNotFoundError" in error and "performance.json" in error and "batch_03" in error
    assert not out.exists()


def test_inconsistent_manifest_is_rejected(tmp_path):
    write_reports(tmp_path)
    path = tmp_path / "batch_results.json"
    manifest = json.loads(path.read_text())
    manifest[0]["drifted"] = 0
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="disagrees.*batch_01"):
        sensitivity.read_analysis(tmp_path)
