import json

import pytest

from src.monitoring.retrain_trigger import decide, main


@pytest.mark.parametrize("share,recall,auc,reasons", [
    (0.29, 0.827, 0.85, []),
    (0.30, 0.777, 0.80, ["data drift"]),
    (0.0, 0.776, 0.80, ["recall degradation"]),
    (0.0, 0.777, 0.799, ["ROC-AUC below quality-gate floor"]),
    (1.0, 0.1, 0.2, ["data drift", "recall degradation", "ROC-AUC below quality-gate floor"]),
])
def test_rule_boundaries(share, recall, auc, reasons):
    result = decide({"share_drifted_features": share}, {"recall": recall, "roc_auc": auc})
    assert result["reasons"] == reasons
    assert result["retrain_needed"] == bool(reasons)


@pytest.mark.parametrize("bad", [None, True, "0.9", float("nan"), float("inf"), -1, 2])
def test_rejects_unusable_metrics(bad):
    with pytest.raises(ValueError):
        decide({"share_drifted_features": 0}, {"recall": bad, "roc_auc": 0.8})


def test_rejects_different_batch_identity():
    with pytest.raises(ValueError, match="different"):
        decide({"share_drifted_features": 0, "data_sha256": "a"},
               {"recall": 0.9, "roc_auc": 0.9, "data_sha256": "b"})


def test_missing_input_exits_two_and_clears_stale_decision(tmp_path):
    output = tmp_path / "trigger.json"
    output.write_text('{"retrain_needed": true}')
    assert main(["--drift", str(tmp_path / "missing"), "--out", str(output)]) == 2
    assert not output.exists()


def test_decision_is_exit_zero_even_when_retraining_needed(tmp_path):
    drift, performance, output = (tmp_path / name for name in ("drift.json", "performance.json", "trigger.json"))
    drift.write_text(json.dumps({"share_drifted_features": 0.5}))
    performance.write_text(json.dumps({"recall": 0.8, "roc_auc": 0.9}))
    assert main(["--drift", str(drift), "--performance", str(performance), "--out", str(output)]) == 0
    assert json.loads(output.read_text())["retrain_needed"] is True
