import json

import numpy as np
import pytest

from scripts import monitor_batch
from tests.test_drift import fixture


def test_workflow_pairs_reports_by_fingerprint_and_removes_stale_decision(tmp_path, monkeypatch):
    model, frame = fixture()
    model.load = lambda: None
    model.model_info = {"decision_threshold": 0.5}
    frame["cardiometabolic"] = np.tile([0, 1], len(frame) // 2)
    source, out = tmp_path / "batch.csv", tmp_path / "reports"
    frame.to_csv(source, index=False)
    monkeypatch.setattr(monitor_batch, "Predictor", lambda: model)
    args = ["--current", str(source), "--reference", str(source), "--out", str(out)]
    assert monitor_batch.main(args) == 0
    reports = [json.loads((out / name).read_text()) for name in ("drift_summary.json", "performance.json", "trigger.json")]
    assert len({report["data_sha256"] for report in reports}) == 1
    assert reports[0]["share_drifted_features"] == 0
    source.unlink()
    with pytest.raises(FileNotFoundError):
        monitor_batch.main(args)
    assert not (out / "trigger.json").exists()
