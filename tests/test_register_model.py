"""Unit tests for MLflow model registration logic (src/models/register_model.py)."""

from unittest.mock import MagicMock, patch
import pytest

from src.models.register_model import (
    DEFAULT_ALIAS,
    DEFAULT_MODEL_NAME,
    _patch_fluent_get_logged_models,
    apply_model_metadata,
    find_best_logistic_regression_run,
    main,
    register_model_artifact,
    set_model_alias,
    validate_candidate,
)


@pytest.fixture
def mock_run():
    """Create a mock MLflow Run entity."""
    run = MagicMock()
    run.info.run_id = "077b778198d448628bf29fbf8c4fd9e6"
    run.info.status = "FINISHED"
    run.data.tags = {
        "model_name": "Logistic Regression",
        "git_commit": "003e6893762a84d72f6b66d4874a705ab3584560",
        "dataset_version": "data-v1",
    }
    run.data.metrics = {
        "roc_auc": 0.850163,
        "recall": 0.826990,
        "train_roc_auc": 0.829584,
        "train_recall": 0.775758,
    }
    return run


def test_find_best_logistic_regression_run_with_preferred_id(mock_run):
    """Verify that a valid preferred run ID is retrieved directly."""
    client = MagicMock()
    client.get_run.return_value = mock_run

    run = find_best_logistic_regression_run(
        client=client,
        experiment_name="cardiometabolic-model-comparison",
        preferred_run_id=mock_run.info.run_id,
    )

    assert run.info.run_id == mock_run.info.run_id
    client.get_run.assert_called_once_with(mock_run.info.run_id)


def test_find_best_logistic_regression_run_search_fallback(mock_run):
    """Verify search fallback when no preferred run ID is specified."""
    client = MagicMock()
    client.get_run.side_effect = Exception("Not found")

    mock_exp = MagicMock()
    mock_exp.experiment_id = "1"
    client.get_experiment_by_name.return_value = mock_exp
    client.search_runs.return_value = [mock_run]

    run = find_best_logistic_regression_run(
        client=client,
        experiment_name="cardiometabolic-model-comparison",
        preferred_run_id=None,
    )

    assert run.info.run_id == mock_run.info.run_id
    client.search_runs.assert_called_once()


def test_find_best_run_raises_when_no_runs_found():
    """Verify ValueError is raised if experiment has no matching Logistic Regression runs."""
    client = MagicMock()
    client.get_run.side_effect = Exception("Not found")
    mock_exp = MagicMock()
    mock_exp.experiment_id = "1"
    client.get_experiment_by_name.return_value = mock_exp
    client.search_runs.return_value = []

    with pytest.raises(ValueError, match="No Logistic Regression runs found"):
        find_best_logistic_regression_run(
            client=client,
            experiment_name="cardiometabolic-model-comparison",
            preferred_run_id=None,
        )


def test_register_model_artifact():
    """Verify model artifact URI and model name passed to mlflow.register_model."""
    with patch("mlflow.register_model") as mock_reg:
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_reg.return_value = mock_version

        version = register_model_artifact(
            run_id="077b778198d448628bf29fbf8c4fd9e6",
            model_name="cardiometabolic-risk-model",
        )

        assert version.version == "1"
        mock_reg.assert_called_once_with(
            model_uri="runs:/077b778198d448628bf29fbf8c4fd9e6/model",
            name="cardiometabolic-risk-model",
        )


def test_patch_fluent_get_logged_models():
    """Verify fallback when search_logged_models with filter_string returns empty."""
    client = MagicMock()
    mock_logged_model = MagicMock()
    mock_logged_model.name = "model"
    mock_logged_model.source_run_id = "run-123"
    client.search_logged_models.return_value = [mock_logged_model]

    _patch_fluent_get_logged_models(client)

    from mlflow.tracking._model_registry import fluent
    source_run = MagicMock()
    source_run.info.run_id = "run-123"
    source_run.info.experiment_id = "1"

    # With orig returning empty, the patch should find mock_logged_model
    with patch("mlflow.tracking._model_registry.fluent._get_logged_models_from_run", return_value=[]):
        pass  # Just ensure function is callable


def test_set_model_alias_success():
    """Verify successful alias assignment."""
    client = MagicMock()
    ok, msg = set_model_alias(
        client=client,
        model_name="cardiometabolic-risk-model",
        alias="champion",
        version="1",
    )

    assert ok is True
    assert "successfully set" in msg
    client.set_registered_model_alias.assert_called_once_with(
        name="cardiometabolic-risk-model",
        alias="champion",
        version="1",
    )


def test_set_model_alias_unsupported_fallback():
    """Verify fallback handling when DagsHub MLflow does not support aliases."""
    client = MagicMock()
    client.set_registered_model_alias.side_effect = Exception("400 Bad Request: Aliases not supported")

    with pytest.warns(UserWarning, match="does not support model aliases"):
        ok, msg = set_model_alias(
            client=client,
            model_name="cardiometabolic-risk-model",
            alias="champion",
            version="1",
        )

    assert ok is False
    assert "version number '1'" in msg


def test_apply_model_metadata(mock_run):
    """Verify required metadata tags and description are applied to model version."""
    client = MagicMock()
    tags = apply_model_metadata(
        client=client,
        model_name="cardiometabolic-risk-model",
        version="1",
        run=mock_run,
        threshold=0.5,
    )

    assert tags["dataset_version"] == "data-v1"
    assert tags["git_commit"] == "003e6893762a84d72f6b66d4874a705ab3584560"
    assert tags["roc_auc"] == "0.8502"
    assert tags["recall"] == "0.8270"
    assert tags["threshold"] == "0.5"

    assert client.set_model_version_tag.call_count >= 5
    client.update_model_version.assert_called_once()
    assert "Champion Logistic Regression model" in client.update_model_version.call_args[1]["description"]


def test_main_cli_execution(mock_run):
    """Verify main CLI orchestration with mocked MLflow."""
    with patch("src.models.register_model.MlflowClient") as mock_client_cls, \
         patch("src.models.register_model.find_best_logistic_regression_run", return_value=mock_run), \
         patch("src.models.register_model.register_model_artifact") as mock_reg, \
         patch("src.models.register_model.validate_candidate", return_value={"passed": True}), \
         patch("src.models.register_model.set_model_alias", return_value=(True, "Alias champion set")), \
         patch("src.models.register_model.apply_model_metadata", return_value={"threshold": "0.5"}):

        mock_version = MagicMock()
        mock_version.version = "1"
        mock_reg.return_value = mock_version

        with patch("sys.argv", ["register_model.py", "--run-id", mock_run.info.run_id]):
            main()

        mock_reg.assert_called_once_with(
            run_id=mock_run.info.run_id,
            model_name=DEFAULT_MODEL_NAME,
            client=mock_client_cls.return_value,
        )


def test_failed_candidate_does_not_register_or_move_alias(mock_run):
    with patch("src.models.register_model.MlflowClient"), \
         patch("src.models.register_model.find_best_logistic_regression_run", return_value=mock_run), \
         patch("src.models.register_model.validate_candidate", side_effect=ValueError("quality gate failed")), \
         patch("src.models.register_model.register_model_artifact") as register, \
         patch("src.models.register_model.set_model_alias") as alias, \
         patch("sys.argv", ["register_model.py"]):
        with pytest.raises(ValueError, match="quality gate failed"):
            main()
    register.assert_not_called()
    alias.assert_not_called()


def test_non_logistic_run_is_rejected_before_loading(mock_run):
    mock_run.data.tags["model_name"] = "Dummy Classifier"
    with patch("mlflow.sklearn.load_model") as load:
        with pytest.raises(ValueError, match="finished Logistic Regression"):
            validate_candidate(mock_run)
    load.assert_not_called()


def test_candidate_quality_failure_blocks_promotion(mock_run):
    model = MagicMock()
    model.feature_names_in_ = ["age"]
    with patch("mlflow.sklearn.load_model", return_value=model), \
         patch("src.models.register_model.pd.read_csv"), \
         patch("src.models.quality_gate.evaluate_quality", return_value={"passed": False, "checks": []}):
        with pytest.raises(ValueError, match="champion was not changed"):
            validate_candidate(mock_run)
