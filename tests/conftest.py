"""Shared, offline fixtures for champion-model quality checks."""

from pathlib import Path

import pandas as pd
import pytest

from src.api.predictor import Predictor

ROOT = Path(__file__).resolve().parents[1]


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
