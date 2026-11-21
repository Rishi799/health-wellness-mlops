import json

import numpy as np
import pandas as pd
import pytest

from src.models.retrain import load_trigger, prepare_splits, train_challenger


def rows(start, n):
    return pd.DataFrame({"seqn": range(start, start + n), "cardiometabolic": np.tile([0, 1], n // 2)})


def test_splits_are_disjoint_and_reproducible():
    old_train, old_test, production = rows(1, 20), rows(100, 10), rows(1000, 50)
    train, new_train, new_test = prepare_splits(old_train, old_test, production)
    assert len(train) == 60 and len(new_test) == 10
    assert not set(train.seqn) & set(new_test.seqn)
    assert not set(train.seqn) & set(old_test.seqn)
    again = prepare_splits(old_train, old_test, production)
    for left, right in zip((train, new_train, new_test), again):
        pd.testing.assert_frame_equal(left, right)


def test_overlapping_participants_rejected():
    with pytest.raises(ValueError, match="overlap"):
        prepare_splits(rows(1, 20), rows(1, 10), rows(1000, 50))


def test_trigger_false_does_not_request_training(tmp_path):
    path = tmp_path / "trigger.json"
    path.write_text(json.dumps({"retrain_needed": False, "reasons": []}))
    assert not load_trigger(path)["retrain_needed"]
    assert load_trigger(path, force=True)["retrain_needed"]
    path.write_text('{"retrain_needed": "true"}')
    with pytest.raises(ValueError):
        load_trigger(path)
    with pytest.raises(FileNotFoundError):
        load_trigger(tmp_path / "missing")


def test_training_retains_contract_and_settings():
    from src.api.schemas import PredictionInput
    features = list(PredictionInput.model_fields)
    frame = pd.DataFrame({name: np.arange(40) % 3 + 1.0 for name in features})
    frame["smoking_status"], frame["phq9_category"] = "never", "minimal"
    frame["cardiometabolic"] = np.tile([0, 1], 20)
    first, second = train_challenger(frame, features), train_challenger(frame, features)
    assert list(first.feature_names_in_) == features
    assert first["classifier"].class_weight == "balanced"
    np.testing.assert_array_equal(first.predict_proba(frame[features]), second.predict_proba(frame[features]))
    with pytest.raises(ValueError):
        train_challenger(frame, features + ["seqn"])
