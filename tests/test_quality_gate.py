"""Exercise gate policy offline, including rejection and report-writing paths."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.models import quality_gate as gate


class FixedProbabilityModel:
    """Controlled probabilities let tests distinguish warning and failure policy."""

    classes_ = np.array([0, 1])

    def __init__(self, features, positive=0.8, negative=0.2):
        self.feature_names_in_ = np.array(features)
        self.positive = positive
        self.negative = negative

    def predict_proba(self, frame):
        probability = np.where(frame["sbp"] > 120, self.positive, self.negative)
        return np.column_stack([1 - probability, probability])


@pytest.fixture
def synthetic_case():
    info = json.loads((gate.REPO_ROOT / "models/model_info.json").read_text(encoding="utf-8"))
    features = info["features"]
    y = np.tile([0, 0, 1, 1], 10)
    frame = pd.DataFrame({name: np.zeros(40) for name in features})
    frame["age"] = np.repeat([25, 45, 65, 80], 10)
    frame["sex"] = np.tile([1, 2], 20)
    frame["sbp"] = np.where(y, 140, 110)
    frame[gate.TARGET] = y
    frame["seqn"] = np.arange(40)
    predictor = SimpleNamespace(features=features, model_info=deepcopy(info),
                                model_version=info["model_version"],
                                model=FixedProbabilityModel(features), load=lambda: None)
    return predictor, frame


def _run_main(monkeypatch, tmp_path, predictor, frame):
    train_path, test_path = tmp_path / "train.csv", tmp_path / "test.csv"
    frame.to_csv(train_path, index=False)
    frame.to_csv(test_path, index=False)
    monkeypatch.setattr(gate, "Predictor", lambda: predictor)
    monkeypatch.setattr(gate, "TRAIN_PATH", train_path)
    monkeypatch.setattr(gate, "TEST_PATH", test_path)
    report_path = tmp_path / "model_quality.json"
    plot_path = tmp_path / "calibration.png"
    code = gate.main(["--report", str(report_path), "--calibration-plot", str(plot_path)])
    return code, json.loads(report_path.read_text(encoding="utf-8")), plot_path


def test_real_champion_passes_and_writes_report(champion, model_train, model_test, tmp_path):
    report = gate.evaluate_quality(champion, model_train, model_test, tmp_path / "calibration.png")
    assert report["passed"], report["checks"]
    assert report["values"]["roc_auc"] >= 0.80
    assert report["values"]["recall"] >= 0.75
    assert len(report["subgroups"]) == 6
    assert (tmp_path / "calibration.png").is_file()
    assert report["calibration_plot"] == "calibration.png"


@pytest.mark.real_artifacts
def test_default_paths_store_relative_calibration_plot(monkeypatch, tmp_path):
    # Report generation in a test must never rewrite committed review evidence.
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(gate, "REPORT_PATH", tmp_path / "reports/model_quality.json")
    monkeypatch.setattr(gate, "PLOT_PATH", tmp_path / "reports/figures/calibration.png")
    assert gate.main([]) == 0
    report_text = gate.REPORT_PATH.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["calibration_plot"] == "reports/figures/calibration.png"
    assert ":" not in report["calibration_plot"]
    assert "\\" not in report_text
    assert "C:" not in report_text


def test_main_writes_json_and_calibration(monkeypatch, tmp_path, synthetic_case):
    code, report, plot = _run_main(monkeypatch, tmp_path, *synthetic_case)
    assert code == 0
    assert report["passed"]
    assert report["thresholds"] == gate.RULES
    assert report["model_name"] == "cardiometabolic-risk-model"
    assert report["model_version"] == "2"
    assert report["date"]
    assert plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert sum(g["n"] for g in report["subgroups"] if g["field"] == "age") == 40


def test_dummy_model_fails_with_exit_one(monkeypatch, tmp_path, synthetic_case):
    predictor, frame = synthetic_case
    predictor.model = DummyClassifier(strategy="most_frequent").fit(frame[predictor.features], frame[gate.TARGET])
    code, report, _ = _run_main(monkeypatch, tmp_path, predictor, frame)
    assert code == 1
    assert not report["passed"]
    failed = {check["check"] for check in report["checks"] if check["status"] == "FAIL"}
    assert {"roc_auc", "recall", "auc_improvement"} <= failed
    assert report["values"]["roc_auc"] == report["values"]["dummy_auc"] == 0.5


def test_contract_failures_are_reported_together(synthetic_case, tmp_path):
    predictor, frame = synthetic_case
    predictor.features = [*predictor.features, "seqn"]
    predictor.model.feature_names_in_ = np.array(predictor.features)
    report = gate.evaluate_quality(predictor, frame, frame, tmp_path / "calibration.png")
    failed = {c["check"] for c in report["checks"] if c["status"] == "FAIL"}
    assert {"no_leakage", "input_contract"} <= failed
    assert report["values"]["roc_auc"] == 1


def test_brier_warning_does_not_fail_gate(synthetic_case, tmp_path):
    predictor, frame = synthetic_case
    predictor.model = FixedProbabilityModel(predictor.features, positive=0.51, negative=0.49)
    report = gate.evaluate_quality(predictor, frame, frame, tmp_path / "calibration.png")
    assert report["passed"]
    assert report["values"]["brier_score"] > 0.20
    assert next(c for c in report["checks"] if c["check"] == "brier_score")["status"] == "WARN"


def test_low_subgroup_recall_is_only_warning(synthetic_case, tmp_path):
    predictor, frame = synthetic_case
    original = predictor.model.predict_proba

    def probabilities(rows):
        result = original(rows)
        result[rows["age"].eq(25) & rows["sbp"].gt(120)] = [0.51, 0.49]
        return result

    predictor.model.predict_proba = probabilities
    report = gate.evaluate_quality(predictor, frame, frame, tmp_path / "calibration.png")
    assert report["passed"]
    group = next(g for g in report["subgroups"] if g["group"] == "20-39")
    assert group["n"] == 10
    assert group["positives"] == 4
    assert group["recall"] == 0
    assert next(c for c in report["checks"] if c["check"] == "age=20-39 recall")["status"] == "WARN"


def test_empty_and_single_class_subgroups_are_explicit(synthetic_case, tmp_path):
    predictor, frame = synthetic_case
    frame["age"] = 25
    frame.loc[frame["sex"].eq(1), gate.TARGET] = 0
    report = gate.evaluate_quality(predictor, frame, frame, tmp_path / "calibration.png")
    empty = next(g for g in report["subgroups"] if g["group"] == "80")
    assert empty == {"field": "age", "group": "80", "n": 0, "positives": 0,
                     "recall": None, "precision": None, "roc_auc": None}
    male = next(g for g in report["subgroups"] if g["field"] == "sex" and g["group"] == "1")
    assert male["n"] == 20
    assert male["recall"] is None
    assert male["roc_auc"] is None


def test_threshold_is_read_from_metadata(synthetic_case, tmp_path):
    predictor, frame = synthetic_case
    predictor.model_info["decision_threshold"] = 0.9
    report = gate.evaluate_quality(predictor, frame, frame, tmp_path / "calibration.png")
    assert report["decision_threshold"] == 0.9
    assert report["values"]["recall"] == 0
    assert not report["passed"]


def test_bad_probabilities_fail_gate(synthetic_case, tmp_path):
    predictor, frame = synthetic_case
    predictor.model.predict_proba = lambda rows: np.full((len(rows), 2), np.nan)
    report = gate.evaluate_quality(predictor, frame, frame, tmp_path / "calibration.png")
    assert not report["passed"]
    assert any(c["check"] == "evaluation" and c["status"] == "FAIL" for c in report["checks"])


def test_artifact_metadata_mismatch_fails(synthetic_case, tmp_path):
    predictor, frame = synthetic_case
    predictor.model.feature_names_in_ = predictor.model.feature_names_in_[::-1]
    report = gate.evaluate_quality(predictor, frame, frame, tmp_path / "calibration.png")
    assert not report["passed"]
    assert next(c for c in report["checks"] if c["check"] == "artifact_features")["status"] == "FAIL"


def test_missing_inputs_write_all_errors(monkeypatch, tmp_path):
    def fail_load(self):
        raise FileNotFoundError("champion missing")

    monkeypatch.setattr(gate.Predictor, "load", fail_load)
    monkeypatch.setattr(gate, "TRAIN_PATH", tmp_path / "missing_train.csv")
    monkeypatch.setattr(gate, "TEST_PATH", tmp_path / "missing_test.csv")
    path = tmp_path / "quality.json"
    assert gate.main(["--report", str(path)]) == 1
    report = json.loads(path.read_text(encoding="utf-8"))
    assert not report["passed"]
    assert len(report["checks"]) == 3
