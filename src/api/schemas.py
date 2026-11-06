"""Pydantic schemas for cardiometabolic risk inference API (Phase 2 Task D).

All feature types, ranges, and allowed categories strictly adhere to
the project data dictionary and src/data/validate.py pandera schemas.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionInput(BaseModel):
    """Input payload representing 21 patient features.

    Demographic identifiers (age, sex) are required.
    All clinical measurements, lifestyle habits, and mental health indicators
    are optional and default to None, which the pipeline's imputation steps
    gracefully resolve.
    """

    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def reject_boolean_values(cls, value):
        # bool is a subclass of int in Python, but is not a survey category or measurement.
        if isinstance(value, bool):
            raise ValueError("Use a numeric value or category, not a boolean")
        return value

    # Demographics (Required)
    age: int = Field(
        ...,
        ge=20,
        le=80,
        description="Age in years (adult cohort 20–80 per validate.py)",
        examples=[45],
    )
    sex: int = Field(
        ...,
        ge=1,
        le=2,
        description="Biological sex: 1 = Male, 2 = Female",
        examples=[1],
    )

    # Demographics & Socioeconomic (Optional)
    education: Optional[Literal[1.0, 2.0, 3.0, 4.0, 5.0]] = Field(
        None,
        description="Education level: 1 = <9th grade, 2 = 9-11th grade, 3 = High school/GED, 4 = Some college/AA, 5 = College grad+",
        examples=[3.0],
    )
    income_ratio: Optional[float] = Field(
        None,
        ge=0.0,
        le=5.0,
        description="Family income-to-poverty ratio (0.0–5.0)",
        examples=[2.5],
    )

    # Body & Clinical Measurements (Optional)
    height_cm: Optional[float] = Field(
        None,
        ge=50.0,
        le=230.0,
        description="Standing height in cm (50.0–230.0)",
        examples=[175.0],
    )
    weight_kg: Optional[float] = Field(
        None,
        ge=2.0,
        le=300.0,
        description="Weight in kg (2.0–300.0)",
        examples=[78.5],
    )
    bmi: Optional[float] = Field(
        None,
        ge=12.0,
        le=80.0,
        description="Body Mass Index in kg/m² (12.0–80.0)",
        examples=[25.6],
    )
    waist_cm: Optional[float] = Field(
        None,
        ge=30.0,
        le=200.0,
        description="Waist circumference in cm (30.0–200.0)",
        examples=[88.0],
    )
    sbp: Optional[float] = Field(
        None,
        ge=60.0,
        le=260.0,
        description="Systolic blood pressure in mmHg (60.0–260.0)",
        examples=[120.0],
    )
    dbp: Optional[float] = Field(
        None,
        ge=20.0,
        le=150.0,
        description="Diastolic blood pressure in mmHg (20.0–150.0)",
        examples=[80.0],
    )
    pulse: Optional[float] = Field(
        None,
        ge=30.0,
        le=200.0,
        description="Resting pulse in beats per minute (30.0–200.0)",
        examples=[72.0],
    )
    told_high_bp: Optional[Literal[0.0, 1.0]] = Field(
        None,
        description="Ever told by doctor had high blood pressure: 1 = Yes, 0 = No",
        examples=[0.0],
    )
    total_chol: Optional[float] = Field(
        None,
        ge=50.0,
        le=600.0,
        description="Total serum cholesterol in mg/dL (50.0–600.0)",
        examples=[195.0],
    )
    hdl: Optional[float] = Field(
        None,
        ge=10.0,
        le=200.0,
        description="Direct HDL cholesterol in mg/dL (10.0–200.0)",
        examples=[55.0],
    )

    # Lifestyle (Optional)
    smoking_status: Optional[Literal["never", "former", "current"]] = Field(
        None,
        description="Cigarette smoking status ('never', 'former', 'current')",
        examples=["never"],
    )
    alcohol_drinks_week: Optional[float] = Field(
        None,
        ge=0.0,
        le=110.0,
        description="Alcoholic drinks per week (0.0–110.0)",
        examples=[2.0],
    )
    exercise_min_week: Optional[float] = Field(
        None,
        ge=0.0,
        le=3000.0,
        description="Moderate-to-vigorous physical activity minutes per week (0.0–3000.0)",
        examples=[150.0],
    )
    sedentary_min_day: Optional[float] = Field(
        None,
        ge=0.0,
        le=1440.0,
        description="Sedentary time in minutes per day (0.0–1440.0)",
        examples=[360.0],
    )
    sleep_hours: Optional[float] = Field(
        None,
        ge=2.0,
        le=14.0,
        description="Typical sleep duration in hours (2.0–14.0)",
        examples=[7.5],
    )

    # Mental Health (Optional)
    phq9_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=27.0,
        description="Total PHQ-9 depression screening score (0.0–27.0)",
        examples=[3.0],
    )
    phq9_category: Optional[
        Literal["minimal", "mild", "moderate", "moderately severe", "severe"]
    ] = Field(
        None,
        description="PHQ-9 severity category: 'minimal', 'mild', 'moderate', 'moderately severe', 'severe'",
        examples=["minimal"],
    )


class PredictionResponse(BaseModel):
    """Inference response containing prediction metrics and clinical disclaimer."""

    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted probability of cardiometabolic risk (0.0–1.0)",
    )
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Risk score on a 0–100 percentage scale",
    )
    risk_band: Literal["Low", "Moderate", "High"] = Field(
        ...,
        description="Agreed risk category: Low (< 30), Moderate (30–59), High (>= 60)",
    )
    model_version: str = Field(
        ...,
        description="Registered champion model version",
    )
    disclaimer: str = Field(
        ...,
        description="Informational clinical disclaimer",
    )


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: Literal["healthy", "unhealthy"]
    model_loaded: bool
    model_version: Optional[str] = None
    error: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Model metadata response schema."""

    model_name: str
    model_version: str
    source_run_id: Optional[str] = None
    git_commit: Optional[str] = None
    service_git_commit: str = "unknown"
    features: List[str]
    decision_threshold: float
    metrics: Dict[str, Any]
    test_roc_auc: Optional[float] = None
    test_recall: Optional[float] = None
    export_date: Optional[str] = None

