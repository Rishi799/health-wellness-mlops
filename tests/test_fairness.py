import json
from types import SimpleNamespace

import pytest

from src.monitoring.fairness import subgroup_metrics


def test_challenger_default_assessment_uses_only_holdout(tmp_path, monkeypatch):
    from src.monitoring import fairness
    production = tmp_path / "data/production"
    production.mkdir(parents=True)
    (production / "test_2021.csv").write_text("marker\n1\n")
    (production / "2021-2023_clean.csv").write_text("marker\n2\n")
    artifact = tmp_path / "candidate.skops"
    artifact.write_bytes(b"candidate")
    model = SimpleNamespace(load=lambda: None, model_info={}, model_path=artifact)
    seen = []
    monkeypatch.setattr(fairness, "ROOT", tmp_path)
    monkeypatch.setattr(fairness, "Predictor", lambda *args: model)
    monkeypatch.setattr(fairness, "assess", lambda model, frame: seen.append(frame.marker.tolist()) or {})
    out = tmp_path / "fairness.json"
    assert fairness.main(["--challenger", "--out", str(out)]) == 0
    assert seen == [[1]]
    assert json.loads(out.read_text())["dataset"] == "test_2021.csv"


def test_counts_intervals_and_recall_difference():
    result = subgroup_metrics([1, 1, 0, 1, 0], [0.8, 0.2, 0.1, 0.9, 0.6], ['a', 'a', 'a', 'b', 'b'])
    a, b = result['groups']
    assert a['n'] == 3 and a['positives'] == 2 and a['detected'] == 1
    assert a['recall'] == 0.5 and b['recall'] == 1.0
    assert a['recall_ci95_wilson'][0] < 0.5 < a['recall_ci95_wilson'][1]
    assert result['recall_difference'] == 0.5


def test_no_positives_preserves_unavailable_recall():
    group = subgroup_metrics([0, 0], [0.2, 0.8], ['a', 'a'])['groups'][0]
    assert group['recall'] is None and group['recall_ci95_wilson'] is None


def test_invalid_probabilities_rejected():
    with pytest.raises(ValueError):
        subgroup_metrics([1], [float('nan')], ['a'])
