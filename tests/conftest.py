"""Shared, offline fixtures for champion-model quality checks."""

from pathlib import Path

import pandas as pd
import pytest

from src.api.predictor import Predictor

ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption("--offline", action="store_true",
                     help="Run tests that need no downloaded NHANES data or champion artifact")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--offline"):
        for item in items:
            if (item.get_closest_marker("real_artifacts") or
                    {"champion", "model_train", "model_test"} & set(item.fixturenames)):
                item.add_marker(pytest.mark.skip(reason="requires DVC artifacts; full CI tier"))


@pytest.fixture(scope="session")
def champion():
    predictor = Predictor()
    predictor.load()
    return predictor


@pytest.fixture(scope="session")
def model_train():
    return pd.read_csv(ROOT / "data/processed/train.csv")


@pytest.fixture(scope="session")
def model_test():
    return pd.read_csv(ROOT / "data/processed/test.csv")
