import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.models.retrain import train_challenger
from src.monitoring.explanations import explain


def test_encoded_explanations_reconstruct_and_aggregate_inputs():
    features = json.loads((Path(__file__).resolve().parents[1] / 'models/model_info.json').read_text())['features']
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({name: rng.normal(size=60) for name in features})
    frame['sex'], frame['education'] = np.tile([1, 2], 30), 3.0
    frame['smoking_status'] = np.tile(['never', 'current'], 30)
    frame['phq9_category'] = 'minimal'
    frame['cardiometabolic'] = np.tile([0, 1], 30)
    model = train_challenger(frame, features)
    predictor = SimpleNamespace(model=model, features=features, model_version='test')
    result = explain(predictor, frame, frame, max_rows=20)
    assert result['additivity_max_error'] < 1e-7
    assert result['sample_rows'] == 20
    assert {item['feature'] for item in result['features']} == set(features)
    assert result['units'] == 'positive-class log-odds'
