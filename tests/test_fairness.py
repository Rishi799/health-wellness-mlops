import pytest

from src.monitoring.fairness import subgroup_metrics


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
