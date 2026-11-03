"""Unit tests and artifact verification for model export (Phase 2 Task C).

Tests are designed to be completely independent of pre-existing generated files
by using pytest tmp_path fixtures and mocks.
A separate verification suite explicitly inspects the generated artifacts.
"""

from datetime import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
import skops.io as sio

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.export_model import (
    APPROVED_SKOPS_TRUSTED_TYPES,
    DEFAULT_ALIAS,
    DEFAULT_FALLBACK_VERSION,
    DEFAULT_MODEL_NAME,
    EXPECTED_FEATURES,
    SecurityError,
    create_model_info_dict,
    export_model_info_json,
    export_pipeline_skops,
    get_feature_names,
    load_champion_model,
    load_skops_pipeline,
    verify_prediction_parity,
)


@pytest.fixture
def mock_pipeline():
    """Create a realistic fitted sklearn Pipeline matching project specifications."""
    numeric_features = [
        "age", "income_ratio", "height_cm", "weight_kg", "bmi", "waist_cm",
        "sbp", "dbp", "pulse", "told_high_bp", "total_chol", "hdl",
        "alcohol_drinks_week", "exercise_min_week", "sedentary_min_day",
        "sleep_hours", "phq9_score"
    ]
    categorical_features = ["smoking_status", "phq9_category", "sex", "education"]

    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
    ])

    clf = LogisticRegression(class_weight="balanced", max_iter=100)
    pipeline = Pipeline([("preprocessing", prep), ("classifier", clf)])

    # Fit on synthetic 21-feature DataFrame
    rng = np.random.default_rng(42)
    n = 30
    data = {}
    for col in numeric_features:
        data[col] = rng.uniform(10, 100, size=n)
    data["smoking_status"] = rng.choice(["never", "former", "current"], size=n)
    data["phq9_category"] = rng.choice(["minimal", "mild", "moderate"], size=n)
    data["sex"] = rng.choice([1, 2], size=n)
    data["education"] = rng.choice([1, 2, 3], size=n)

    df = pd.DataFrame(data)[EXPECTED_FEATURES]
    y = rng.choice([0, 1], size=n)
    pipeline.fit(df, y)
    return pipeline


@pytest.fixture
def sample_test_df():
    """Create reproducible test dataframe containing all 21 features."""
    rng = np.random.default_rng(100)
    n = 20
    data = {}
    for col in EXPECTED_FEATURES:
        if col in ["smoking_status", "phq9_category"]:
            data[col] = rng.choice(["never", "former", "current", "minimal", "mild"], size=n)
        elif col in ["sex", "education"]:
            data[col] = rng.choice([1, 2, 3], size=n)
        else:
            data[col] = rng.uniform(20, 80, size=n)
    return pd.DataFrame(data)[EXPECTED_FEATURES]


def test_load_champion_model_alias_success():
    """Verify loading champion model using alias when alias is supported."""
    mock_model = MagicMock()
    mock_mv = MagicMock()
    mock_mv.version = "2"
    mock_mv.run_id = "run-abc-123"
    mock_mv.tags = {
        "git_commit": "003e6893762a84d72f6b66d4874a705ab3584560",
        "roc_auc": "0.8502",
        "recall": "0.8270",
        "threshold": "0.5",
    }

    client = MagicMock()
    client.get_model_version_by_alias.return_value = mock_mv
    client.get_model_version.return_value = mock_mv

    with patch("mlflow.sklearn.load_model", return_value=mock_model) as mock_load:
        model, details = load_champion_model(
            client=client,
            model_name=DEFAULT_MODEL_NAME,
            alias=DEFAULT_ALIAS,
            fallback_version=DEFAULT_FALLBACK_VERSION,
        )

        mock_load.assert_called_once_with(f"models:/{DEFAULT_MODEL_NAME}@{DEFAULT_ALIAS}")
        assert model == mock_model
        assert details["model_version"] == "2"
        assert details["source_run_id"] == "run-abc-123"
        assert details["git_commit"] == "003e6893762a84d72f6b66d4874a705ab3584560"
        assert details["test_roc_auc"] == 0.8502
        assert details["test_recall"] == 0.8270


def test_load_champion_model_alias_failure_fallback_to_version():
    """Requirement 2: If alias loading fails, use registered version 2 explicitly."""
    mock_model = MagicMock()
    mock_mv = MagicMock()
    mock_mv.version = "2"
    mock_mv.run_id = "run-fallback-999"
    mock_mv.tags = {
        "git_commit": "fallback_commit_sha",
        "roc_auc": "0.8502",
        "recall": "0.8270",
        "threshold": "0.5",
    }

    client = MagicMock()
    client.get_model_version_by_alias.side_effect = Exception("Alias endpoint unsupported")
    client.get_model_version.return_value = mock_mv

    def load_side_effect(uri):
        if "@champion" in uri:
            raise Exception("400 Bad Request: Aliases not found")
        return mock_model

    with patch("mlflow.sklearn.load_model", side_effect=load_side_effect) as mock_load:
        with pytest.warns(UserWarning, match="Falling back to explicit registered version '2'"):
            model, details = load_champion_model(
                client=client,
                model_name=DEFAULT_MODEL_NAME,
                alias="champion",
                fallback_version="2",
            )

        assert mock_load.call_count == 2
        assert model == mock_model
        assert details["model_version"] == "2"
        assert details["load_source"] == f"models:/{DEFAULT_MODEL_NAME}/2"


def test_get_feature_names_order(mock_pipeline):
    """Verify 21 feature names are extracted in the exact order."""
    features = get_feature_names(mock_pipeline)
    assert len(features) == 21
    assert features == EXPECTED_FEATURES


def test_export_pipeline_skops_and_reload(mock_pipeline, tmp_path):
    """Verify exporting to skops and reloading with approved trusted types."""
    skops_file = tmp_path / "model.skops"
    export_pipeline_skops(mock_pipeline, skops_file)
    assert skops_file.exists()
    assert skops_file.stat().st_size > 0

    # Load with approved types
    loaded = load_skops_pipeline(skops_file, trusted_types=APPROVED_SKOPS_TRUSTED_TYPES)
    assert hasattr(loaded, "predict")
    assert hasattr(loaded, "predict_proba")


def test_load_skops_security_error_on_unapproved_types(tmp_path):
    """Requirement adjustment 1: Disallow unapproved untrusted types and raise SecurityError."""
    skops_file = tmp_path / "dummy.skops"
    sio.dump(np.dtype("float64"), skops_file)

    # Empty whitelist should reject numpy.dtype
    with pytest.raises(SecurityError, match="Security policy violation"):
        load_skops_pipeline(skops_file, trusted_types=[])


def test_model_info_json_schema_and_export(tmp_path):
    """Requirement 4: Verify all required fields in model_info.json."""
    details = {
        "model_name": "cardiometabolic-risk-model",
        "model_version": "2",
        "source_run_id": "077b778198d448628bf29fbf8c4fd9e6",
        "git_commit": "003e6893762a84d72f6b66d4874a705ab3584560",
        "test_roc_auc": 0.850163,
        "test_recall": 0.826990,
        "threshold": 0.5,
    }
    info_dict = create_model_info_dict(details, EXPECTED_FEATURES, export_timestamp="2026-11-03T19:00:00Z")

    assert info_dict["model_name"] == "cardiometabolic-risk-model"
    assert info_dict["model_version"] == "2"
    assert info_dict["source_run_id"] == "077b778198d448628bf29fbf8c4fd9e6"
    assert info_dict["git_commit"] == "003e6893762a84d72f6b66d4874a705ab3584560"
    assert info_dict["features"] == EXPECTED_FEATURES
    assert len(info_dict["features"]) == 21
    assert info_dict["decision_threshold"] == 0.5
    assert info_dict["test_roc_auc"] == 0.8502
    assert info_dict["test_recall"] == 0.8270
    assert info_dict["export_date"] == "2026-11-03T19:00:00Z"

    # Export to json and verify parsing
    json_path = tmp_path / "model_info.json"
    export_model_info_json(info_dict, json_path)
    assert json_path.exists()

    with open(json_path) as f:
        loaded = json.load(f)
    assert loaded == info_dict


def test_verify_prediction_parity_complete_and_missing(mock_pipeline, sample_test_df, tmp_path):
    """Requirement adjustment 2: Parity across complete AND partially missing inputs in 21-feature order."""
    skops_file = tmp_path / "model.skops"
    export_pipeline_skops(mock_pipeline, skops_file)
    exported_model = load_skops_pipeline(skops_file)

    parity = verify_prediction_parity(
        original_model=mock_pipeline,
        exported_model=exported_model,
        test_df=sample_test_df,
        features=EXPECTED_FEATURES,
    )

    assert "complete_inputs" in parity
    assert "partially_missing_inputs" in parity
    assert parity["complete_inputs"]["predictions_identical"] is True
    assert parity["complete_inputs"]["probabilities_allclose"] is True
    assert parity["partially_missing_inputs"]["predictions_identical"] is True
    assert parity["partially_missing_inputs"]["probabilities_allclose"] is True
    assert parity["complete_inputs"]["max_probability_abs_diff"] < 1e-7
    assert parity["partially_missing_inputs"]["max_probability_abs_diff"] < 1e-7


def test_parity_fails_on_mismatched_model(mock_pipeline, sample_test_df):
    """Ensure verify_prediction_parity strictly raises AssertionError on mismatch."""
    mismatched_pipeline = MagicMock()
    mismatched_pipeline.predict.return_value = np.zeros(len(sample_test_df))
    mismatched_pipeline.predict_proba.return_value = np.zeros((len(sample_test_df), 2))

    with pytest.raises(AssertionError, match="Prediction parity failed"):
        verify_prediction_parity(
            original_model=mock_pipeline,
            exported_model=mismatched_pipeline,
            test_df=sample_test_df,
            features=EXPECTED_FEATURES,
        )


# Separate validation for actual exported artifacts in models/
def test_disk_exported_artifacts_validation():
    """Verify the actual generated models/model.skops and models/model_info.json if present."""
    root = Path(__file__).resolve().parents[1]
    skops_file = root / "models/model.skops"
    info_file = root / "models/model_info.json"

    if not (skops_file.exists() and info_file.exists()):
        pytest.skip("Exported artifacts not yet created on disk. Skipping artifact verification.")

    assert skops_file.stat().st_size > 0

    with open(info_file, "r", encoding="utf-8") as f:
        info = json.load(f)

    assert info["model_name"] == DEFAULT_MODEL_NAME
    assert info["model_version"] in ["1", "2"]
    assert len(info["features"]) == 21
    assert info["features"] == EXPECTED_FEATURES
    assert "source_run_id" in info and info["source_run_id"]
    assert "git_commit" in info and info["git_commit"]
    assert "decision_threshold" in info
    assert "test_roc_auc" in info
    assert "test_recall" in info
    assert "export_date" in info

    # Verify reload of disk skops
    loaded = load_skops_pipeline(skops_file, trusted_types=APPROVED_SKOPS_TRUSTED_TYPES)
    assert hasattr(loaded, "predict")
    assert hasattr(loaded, "predict_proba")

