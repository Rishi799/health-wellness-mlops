import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.api.schemas import PredictionInput
from src.monitoring.drift import evaluate_drift, read_inputs


class DriftModel:
    classes_ = [0, 1]

    def predict_proba(self, frame):
        probability = 1 / (1 + np.exp(-frame['age'].to_numpy() / 10))
        return np.column_stack([1 - probability, probability])


def fixture():
    rng = np.random.default_rng(42)
    features = list(PredictionInput.model_fields)
    frame = pd.DataFrame({name: rng.normal(size=120) for name in features})
    frame['sex'], frame['education'], frame['told_high_bp'] = 1, 3.0, 0.0
    frame['smoking_status'], frame['phq9_category'] = 'never', 'minimal'
    return SimpleNamespace(model=DriftModel(), features=features, model_version='2'), frame


def test_identical_inputs_have_no_drift(tmp_path):
    predictor, frame = fixture()
    snapshot, report = evaluate_drift(predictor, frame, frame.copy())
    assert report['share_drifted_features'] == 0
    assert not report['dataset_drift'] and not report['prediction_drift']
    snapshot.save_html(str(tmp_path / 'report.html'))
    assert 'html' in (tmp_path / 'report.html').read_text(encoding='utf-8').lower()


def test_shifted_inputs_flag_drift_and_keep_feature_denominator():
    predictor, frame = fixture()
    shifted = frame.copy()
    for name in ('age', 'bmi', 'sbp', 'dbp', 'pulse', 'height_cm', 'weight_kg'):
        shifted[name] += 10
    _, report = evaluate_drift(predictor, frame, shifted)
    assert report['dataset_drift']
    assert report['feature_count'] == 21
    assert 'prediction' not in report['drifted_features']
    assert report['share_drifted_features'] == len(report['drifted_features']) / 21


def test_missing_feature_distribution_is_explicitly_unavailable():
    predictor, frame = fixture()
    frame['bmi'] = np.nan
    with pytest.raises(ValueError, match='entirely missing'):
        evaluate_drift(predictor, frame, frame)


def test_equivalent_numeric_category_encodings_do_not_drift():
    predictor, frame = fixture()
    current = frame.copy()
    for column in ("sex", "education", "told_high_bp"):
        current[column] = current[column].astype(float)
    _, report = evaluate_drift(predictor, frame, current)
    assert not report["dataset_drift"]
    assert report["share_drifted_features"] == 0


def test_jsonl_contract_and_model_version(tmp_path):
    predictor, _ = fixture()
    log = tmp_path / 'predictions.jsonl'
    record = {'model_version': '2', 'inputs': PredictionInput(age=55, sex=1).model_dump()}
    log.write_text(json.dumps(record) + '\n')
    assert read_inputs(log, predictor).shape == (1, 21)
    record['model_version'] = '3'
    log.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='different model'):
        read_inputs(log, predictor)
