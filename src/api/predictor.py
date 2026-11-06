"""Predictor service for champion model inference (Phase 2 Task D).

Loads models/model.skops and models/model_info.json using approved skops
trusted-type whitelist, executes inference with exact 21-feature ordering,
and computes probability, risk score, and risk band.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import skops.io as sio

from src.api.schemas import ModelInfoResponse, PredictionInput, PredictionResponse

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models/model.skops"
DEFAULT_INFO_PATH = ROOT / "models/model_info.json"

APPROVED_SKOPS_TRUSTED_TYPES: List[str] = [
    "numpy.dtype",
]

CLINICAL_DISCLAIMER: str = (
    "This tool is for informational and educational purposes only and does not "
    "constitute medical advice, diagnosis, or treatment. Consult a licensed healthcare "
    "provider for clinical evaluation."
)


def calculate_risk_band(risk_score: float) -> str:
    """Classify risk score into agreed risk bands.

    Agreed definitions:
      - Low: below 30 (< 30.0)
      - Moderate: 30–59 (30.0 <= score < 60.0)
      - High: 60 or above (>= 60.0)

    Boundaries:
      0.29 prob -> score 29.0 -> "Low"
      0.30 prob -> score 30.0 -> "Moderate"
      0.59 prob -> score 59.0 -> "Moderate"
      0.60 prob -> score 60.0 -> "High"
    """
    if risk_score < 30.0:
        return "Low"
    if risk_score < 60.0:
        return "Moderate"
    return "High"


class Predictor:
    """Singleton service managing model loading, validation, and inference."""

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        info_path: Union[str, Path] = DEFAULT_INFO_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.info_path = Path(info_path)
        self.model: Any = None
        self.model_info: Dict[str, Any] = {}
        self.features: List[str] = []
        self.model_version: str = "unknown"
        self.is_loaded: bool = False
        self.load_error: Optional[str] = None

    def load(self) -> None:
        """Load model artifact and metadata using approved skops whitelist."""
        try:
            if not self.info_path.exists():
                raise FileNotFoundError(f"Model metadata not found at {self.info_path}")
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model artifact not found at {self.model_path}")

            with open(self.info_path, "r", encoding="utf-8") as f:
                self.model_info = json.load(f)

            self.features = list(self.model_info.get("features", []))
            self.model_version = str(self.model_info.get("model_version", "unknown"))

            # Security audit for untrusted types
            untrusted = sio.get_untrusted_types(file=self.model_path)
            unapproved = [t for t in untrusted if t not in APPROVED_SKOPS_TRUSTED_TYPES]
            if unapproved:
                raise ValueError(
                    f"Security policy violation: detected unapproved untrusted types in {self.model_path}: {unapproved}. "
                    f"Permitted types: {APPROVED_SKOPS_TRUSTED_TYPES}"
                )

            self.model = sio.load(self.model_path, trusted=APPROVED_SKOPS_TRUSTED_TYPES)
            ModelInfoResponse.model_validate(self.model_info)
            if (len(self.features) != len(set(self.features)) or
                    set(self.features) != set(PredictionInput.model_fields)):
                raise ValueError("Model metadata does not match the serving feature contract")
            if list(self.model.feature_names_in_) != self.features:
                raise ValueError("Model artifact feature ordering differs from metadata")
            if list(self.model.classes_) != [0, 1]:
                raise ValueError("Model classes must be ordered [0, 1]")
            self.is_loaded = True
            # A readable artifact is not necessarily a usable estimator. Keep
            # /health unavailable unless a representative prediction succeeds.
            self.predict(PredictionInput(age=55, sex=1))
            self.load_error = None
        except Exception as exc:
            self.model = None
            self.is_loaded = False
            self.load_error = f"{type(exc).__name__}: {exc}"
            raise

    def predict(self, input_data: PredictionInput) -> PredictionResponse:
        """Run model inference on single-patient input."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError(f"Model is not loaded. Last error: {self.load_error}")

        # Build feature dictionary preserving the exact feature list and order from model_info.json
        input_dict = input_data.model_dump()
        row: Dict[str, Any] = {}
        for feat in self.features:
            val = input_dict.get(feat)
            row[feat] = np.nan if val is None else val

        df = pd.DataFrame([row])[self.features]

        # Calculate predicted probability of positive class (cardiometabolic risk)
        probs = self.model.predict_proba(df)
        if (np.shape(probs) != (1, 2) or not np.isfinite(probs).all() or
                not ((probs >= 0) & (probs <= 1)).all() or
                not np.allclose(np.sum(probs, axis=1), 1)):
            raise ValueError("Model returned invalid probabilities")
        prob_positive = float(probs[0, 1])

        prob_rounded = round(prob_positive, 4)
        risk_score = round(prob_positive * 100.0, 1)
        risk_band = calculate_risk_band(risk_score)

        return PredictionResponse(
            probability=prob_rounded,
            risk_score=risk_score,
            risk_band=risk_band,
            model_version=self.model_version,
            disclaimer=CLINICAL_DISCLAIMER,
        )


predictor = Predictor()

