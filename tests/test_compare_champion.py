from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.models.compare_champion import compare, promotion_decision, verify_provenance
from tests.test_quality_gate import FixedProbabilityModel


def scores():
    return {"2017-2018": {"roc_auc": 0.85, "recall": 0.82},
            "2021-2023": {"roc_auc": 0.83, "recall": 0.80}}


def test_tie_is_eligible_for_review():
    result = promotion_decision(scores(), scores(), True)
    assert result["promote"]
    assert result["review_required"] and not result["champion_alias_changed"]


@pytest.mark.parametrize("cycle,metric,value", [("2017-2018", "roc_auc", 0.799),
                                              ("2021-2023", "recall", 0.79),
                                              ("2021-2023", "roc_auc", 0.819),
                                              ("2021-2023", "roc_auc", None)])
def test_each_metric_rule_rejects(cycle, metric, value):
    challenger = deepcopy(scores())
    challenger[cycle][metric] = value
    assert not promotion_decision(scores(), challenger, True)["promote"]


def test_hard_gate_failure_blocks_otherwise_better_model():
    challenger = scores()
    challenger["2021-2023"].update(recall=0.9, roc_auc=0.9)
    assert not promotion_decision(scores(), challenger, False)["promote"]


def test_exact_auc_tolerance_and_old_floor_pass():
    champion = scores()
    champion["2021-2023"]["roc_auc"] = 0.81
    challenger = scores()
    challenger["2021-2023"]["roc_auc"] = 0.80
    challenger["2017-2018"]["roc_auc"] = 0.80
    assert promotion_decision(champion, challenger, True)["promote"]


def test_comparison_evaluates_both_holdouts_and_existing_gate(tmp_path):
    info = json.loads((Path(__file__).resolve().parents[1] / "models/model_info.json").read_text())
    features = info["features"]
    labels = np.tile([0, 1], 20)
    frame = pd.DataFrame({name: np.zeros(40) for name in features})
    frame["age"], frame["sex"] = 45, np.tile([1, 2], 20)
    frame["sbp"] = np.where(labels, 140, 110)
    frame["cardiometabolic"] = labels
    model = SimpleNamespace(model=FixedProbabilityModel(features), features=features,
                            model_version="test", model_info=info)
    report, decision = compare(model, model, frame, frame, frame, tmp_path)
    assert decision["promote"]
    assert set(report["champion"]) == {"2017-2018", "2021-2023"}
    assert set(report["quality_gates"]) == {"2017-2018", "2021-2023"}
    assert report["challenger"]["2021-2023"]["n"] == 40
    assert (tmp_path / "calibration_2021-2023.png").exists()


def test_stale_artifact_or_changed_evaluation_data_rejected(tmp_path):
    names = ["data/processed/train.csv", "data/processed/test.csv", "data/production/2021-2023_clean.csv",
             "data/production/train_2021.csv", "data/production/test_2021.csv"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original")
    artifact = tmp_path / "candidate.skops"
    artifact.write_bytes(b"model")
    metadata = {"input_sha256": {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in names},
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
    verify_provenance(metadata, artifact, tmp_path)
    artifact.write_bytes(b"different model")
    with pytest.raises(ValueError, match="artifact"):
        verify_provenance(metadata, artifact, tmp_path)
    artifact.write_bytes(b"model")
    (tmp_path / names[-1]).write_bytes(b"different holdout")
    with pytest.raises(ValueError, match="Data changed"):
        verify_provenance(metadata, artifact, tmp_path)
