"""Labelled monitoring uses the same metrics as the existing quality gate."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.monitoring.performance import evaluate_performance


class FixedModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        probability = frame["score"].to_numpy()
        return np.column_stack([1 - probability, probability])


def case():
    predictor = SimpleNamespace(model=FixedModel(), features=["score"], model_version="2",
                                model_info={"decision_threshold": 0.5})
    frame = pd.DataFrame({"score": [0.1, 0.6, 0.4, 0.9], "cardiometabolic": [0, 0, 1, 1],
                          "age": [25, 25, 65, 65], "sex": [1, 2, 1, 2]})
    return predictor, frame


def test_metrics_and_subgroup_counts():
    report = evaluate_performance(*case())
    assert report["recall"] == report["precision"] == 0.5
    assert report["roc_auc"] == 0.75
    assert report["brier"] == pytest.approx(0.185)
    assert report["n"] == 4 and report["positives"] == 2
    young = next(s for s in report["subgroups"] if s["group"] == "20-39")
    assert young["n"] == 2 and young["recall"] is None


@pytest.mark.parametrize("bad", [np.nan, float("inf"), -0.1, 1.1])
def test_rejects_invalid_probabilities(bad):
    predictor, frame = case()
    frame.loc[0, "score"] = bad
    with pytest.raises(ValueError, match="finite"):
        evaluate_performance(predictor, frame)


def test_single_class_keeps_unavailable_auc_explicit():
    predictor, frame = case()
    frame["cardiometabolic"] = 0
    report = evaluate_performance(predictor, frame)
    assert report["roc_auc"] is None and report["recall"] is None


def test_rejects_unknown_labels():
    predictor, frame = case()
    frame.loc[0, "cardiometabolic"] = np.nan
    with pytest.raises(ValueError, match="labels"):
        evaluate_performance(predictor, frame)


def test_failed_performance_rerun_removes_old_report(tmp_path, monkeypatch):
    from src.monitoring import performance
    predictor, _ = case()
    predictor.load = lambda: None
    monkeypatch.setattr(performance, "Predictor", lambda: predictor)
    out = tmp_path / "performance.json"
    out.write_text('{"recall": 1.0}')
    with pytest.raises(FileNotFoundError):
        performance.main(["--current", str(tmp_path / "missing.csv"), "--out", str(out)])
    assert not out.exists()
