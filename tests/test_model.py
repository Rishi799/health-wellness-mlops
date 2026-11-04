"""Offline regression checks for the registered champion, never a retrained model."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import recall_score, roc_auc_score

from src.models.compare_models import EXCLUDE, TARGET


@pytest.fixture(scope="session")
def probabilities(champion, model_test):
    return champion.model.predict_proba(model_test[champion.features])


@pytest.fixture(scope="session")
def median_person(champion, model_train):
    # Use training medians for numbers and modes for categorical features.
    row = {column: (model_train[column].median() if pd.api.types.is_numeric_dtype(model_train[column])
                    else model_train[column].mode().iloc[0]) for column in champion.features}
    return pd.DataFrame([row])[champion.features]


def test_probability_shape(probabilities, model_test):
    assert probabilities.shape == (len(model_test), 2)


def test_probability_range(probabilities):
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_probability_rows_sum_to_one(probabilities):
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)


def test_missing_numeric_values_are_imputed(champion, median_person):
    row = median_person.copy()
    row[["income_ratio", "total_chol", "phq9_score"]] = np.nan
    probability = champion.model.predict_proba(row)
    assert probability.shape == (1, 2)
    assert np.isfinite(probability).all()
    np.testing.assert_allclose(probability.sum(axis=1), 1)


def test_feature_contract(champion):
    assert len(champion.features) == len(set(champion.features)) == 21
    assert not set(champion.features) & EXCLUDE
    assert list(champion.model.feature_names_in_) == champion.features


@pytest.mark.parametrize("feature,lower,higher", [("age", 30, 70), ("told_high_bp", 0, 1)])
def test_directional_probability(champion, median_person, feature, lower, higher):
    rows = pd.concat([median_person, median_person], ignore_index=True)
    rows[feature] = [lower, higher]
    probability = champion.model.predict_proba(rows)[:, 1]
    assert probability[1] > probability[0]


def test_seqn_invariance(champion, median_person):
    assert "seqn" not in champion.features
    rows = pd.concat([median_person, median_person], ignore_index=True)
    rows["seqn"] = [1, 999999]
    # Test the pipeline itself, including its handling of an extra identifier.
    probability = champion.model.predict_proba(rows)
    np.testing.assert_array_equal(probability[0], probability[1])
    np.testing.assert_array_equal(probability[:1], champion.model.predict_proba(median_person))


def test_deterministic_predictions(champion, model_test):
    rows = model_test[champion.features].head(50)
    np.testing.assert_array_equal(champion.model.predict_proba(rows), champion.model.predict_proba(rows))


def test_minimum_performance(champion, model_train, model_test, probabilities):
    y = model_test[TARGET]
    probability = probabilities[:, 1]
    dummy = DummyClassifier(strategy="most_frequent").fit(model_train[champion.features], model_train[TARGET])
    dummy_probability = dummy.predict_proba(model_test[champion.features])[:, 1]
    auc = roc_auc_score(y, probability)
    recall = recall_score(y, probability >= 0.5)
    assert auc >= 0.80
    assert recall >= 0.75
    assert auc > roc_auc_score(y, dummy_probability)
    assert recall > recall_score(y, dummy.predict(model_test[champion.features]))
