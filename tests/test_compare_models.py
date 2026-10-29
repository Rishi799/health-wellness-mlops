"""Unit tests for model comparison pipeline (src/models/compare_models.py).

Tests:
- All 7 expected models (Dummy, Logistic Regression, Decision Tree, Random Forest,
  Gradient Boosting, Extra Trees, XGBoost) are configured with appropriate hyperparameters.
- Model parameter dictionary defines meaningful MLflow logging metadata for all 7 models.
- Preprocessing transformer handles missing values and categoricals without leakage.
- Pipelines successfully train, predict classes, and predict probabilities for all 7 models.
- Evaluation metrics calculate accuracy, precision, recall, f1, and roc_auc within [0, 1].
- Target and clinical diagnostic proxy columns are strictly excluded from features.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.models.compare_models import (
    EXCLUDE,
    TARGET,
    build_preprocessing,
    evaluate_predictions,
    get_model_params,
    get_models,
)


@pytest.fixture
def synthetic_train_test():
    """Create reproducible synthetic train and test DataFrames matching project schemas."""
    rng = np.random.default_rng(42)
    n_train, n_test = 120, 40

    def generate_df(n):
        return pd.DataFrame({
            "seqn": np.arange(1000, 1000 + n),
            "age": rng.integers(20, 80, size=n),
            "bmi": np.where(rng.random(n) < 0.1, np.nan, rng.uniform(18, 40, size=n)),
            "sbp": np.where(rng.random(n) < 0.1, np.nan, rng.uniform(90, 180, size=n)),
            "sex": rng.choice(["male", "female"], size=n),
            "education": rng.choice([1, 2, 3, 4, 5], size=n),
            "phq9_category": rng.choice(["none", "mild", "moderate", np.nan], size=n),
            "smoking_status": rng.choice(["never", "former", "current", np.nan], size=n),
            "hba1c": rng.uniform(4.5, 10.0, size=n),
            "diabetes": rng.choice([0, 1], size=n),
            "heart_failure": rng.choice([0, 1], size=n),
            "cardiometabolic": rng.choice([0, 1], size=n, p=[0.74, 0.26]),
        })

    train = generate_df(n_train)
    test = generate_df(n_test)
    return train, test


def test_models_dictionary_contains_all_seven_models():
    """Verify that all 7 required models are present with the correct classifier types."""
    models = get_models(scale_pos_weight=2.83)
    expected_models = {
        "Dummy Classifier": DummyClassifier,
        "Logistic Regression": LogisticRegression,
        "Decision Tree": DecisionTreeClassifier,
        "Random Forest": RandomForestClassifier,
        "Gradient Boosting": GradientBoostingClassifier,
        "Extra Trees": ExtraTreesClassifier,
        "XGBoost": XGBClassifier,
    }

    assert set(models.keys()) == set(expected_models.keys()), "All 7 models must be configured"

    for name, expected_class in expected_models.items():
        assert isinstance(models[name], expected_class), f"{name} must be an instance of {expected_class.__name__}"


def test_model_params_covers_all_seven_models():
    """Verify that every model has corresponding hyperparameter metadata for MLflow logging."""
    params = get_model_params(scale_pos_weight=2.83)
    models = get_models(scale_pos_weight=2.83)

    assert set(params.keys()) == set(models.keys())

    for name, p in params.items():
        assert "model_type" in p, f"{name} params must specify model_type"
        assert len(p) >= 2, f"{name} must have meaningful hyperparameters logged"


def test_preprocessing_pipeline_fit_transform(synthetic_train_test):
    """Verify preprocessing ColumnTransformer handles missing values and categoricals."""
    train, test = synthetic_train_test
    feature_cols = [c for c in train.columns if c not in EXCLUDE]
    numeric = ["age", "bmi", "sbp"]
    categorical = ["sex", "education", "phq9_category", "smoking_status"]

    prep = build_preprocessing(numeric=numeric, categorical=categorical)
    transformed_train = prep.fit_transform(train[feature_cols])
    transformed_test = prep.transform(test[feature_cols])

    assert transformed_train.shape[0] == len(train)
    assert transformed_test.shape[0] == len(test)
    assert not np.isnan(transformed_train).any(), "Transformed train data must contain no NaNs"
    assert not np.isnan(transformed_test).any(), "Transformed test data must contain no NaNs"


def test_all_seven_models_pipeline_fit_and_predict(synthetic_train_test):
    """Verify full Pipeline fit, predict, and predict_proba execution for all 7 models."""
    train, test = synthetic_train_test
    X_train = train.drop(columns=[c for c in EXCLUDE if c in train.columns])
    y_train = train[TARGET].astype(int)
    X_test = test.drop(columns=[c for c in EXCLUDE if c in test.columns])
    y_test = test[TARGET].astype(int)

    numeric = ["age", "bmi", "sbp"]
    categorical = ["sex", "education", "phq9_category", "smoking_status"]
    prep = build_preprocessing(numeric=numeric, categorical=categorical)

    models = get_models(scale_pos_weight=2.83)

    for name, clf in models.items():
        pipeline = Pipeline([
            ("preprocessing", prep),
            ("classifier", clf),
        ])
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        assert len(preds) == len(X_test), f"{name} predictions length mismatch"
        assert len(probs) == len(X_test), f"{name} probabilities length mismatch"
        assert ((probs >= 0.0) & (probs <= 1.0)).all(), f"{name} probabilities must be in [0, 1]"

        metrics = evaluate_predictions(y_test, preds, probs)
        for metric_name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert metric_name in metrics, f"{metric_name} missing from {name} evaluation"
            assert 0.0 <= metrics[metric_name] <= 1.0, f"{metric_name} out of bounds [0, 1] for {name}"


def test_target_leakage_exclusion():
    """Ensure clinical proxies (e.g. hba1c, diabetes, stroke) are in EXCLUDE set."""
    leakage_cols = {
        "diabetes", "heart_failure", "coronary_heart_disease",
        "angina", "heart_attack", "stroke", "hba1c",
    }
    for col in leakage_cols:
        assert col in EXCLUDE, f"{col} must be in EXCLUDE to prevent target leakage"
