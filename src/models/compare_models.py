"""Compare four classifiers using the shared train/test split and log to DagsHub MLflow."""

import os
from pathlib import Path
import subprocess

from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data/processed/train.csv"
TEST_PATH = ROOT / "data/processed/test.csv"
OUTPUT_PATH = ROOT / "reports/model_comparison.csv"

TARGET = "cardiometabolic"
DATASET_VERSION = "data-v1"
EXCLUDE = {
    "seqn", TARGET, "diabetes", "heart_failure",
    "coronary_heart_disease", "angina", "heart_attack",
    "stroke", "hba1c", "diet_quality",
}


def get_git_commit() -> str:
    """Retrieve current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def main():
    # Load credentials securely from local .env
    load_dotenv(ROOT / ".env")

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "https://dagshub.com/Rishi799/health-wellness-mlops.mlflow",
    )
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "cardiometabolic-four-model-comparison",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    git_commit = get_git_commit()

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train = train.dropna(subset=[TARGET]).copy()
    test = test.dropna(subset=[TARGET]).copy()

    X_train = train.drop(columns=[c for c in EXCLUDE if c in train.columns])
    y_train = train[TARGET].astype(int)
    X_test = test.drop(columns=[c for c in EXCLUDE if c in test.columns])
    y_test = test[TARGET].astype(int)

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train and test feature columns do not match")

    categorical = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numeric = X_train.select_dtypes(include=["number"]).columns.tolist()

    for col in ("sex", "education", "phq9_category", "smoking_status"):
        if col in X_train.columns and col not in categorical:
            categorical.append(col)
            if col in numeric:
                numeric.remove(col)

    # Convert categorical columns consistently in train and test.
    for col in categorical:
        X_train[col] = X_train[col].astype(object).where(
            X_train[col].notna(), np.nan
        )
        X_test[col] = X_test[col].astype(object).where(
            X_test[col].notna(), np.nan
        )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])

    models = {
        "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
    }

    # Explicit hyperparameters to log for each model
    model_params = {
        "Dummy Classifier": {
            "model_type": "DummyClassifier",
            "strategy": "most_frequent",
        },
        "Logistic Regression": {
            "model_type": "LogisticRegression",
            "max_iter": 1000,
            "class_weight": "balanced",
            "solver": "lbfgs",
        },
        "Decision Tree": {
            "model_type": "DecisionTreeClassifier",
            "max_depth": 8,
            "class_weight": "balanced",
            "random_state": 42,
        },
        "Random Forest": {
            "model_type": "RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": "None",
            "class_weight": "balanced",
            "random_state": 42,
        },
    }

    rows = []
    for name, classifier in models.items():
        with mlflow.start_run(run_name=name):
            # Set tags: Git commit and dataset version data-v1
            mlflow.set_tags({
                "dataset_version": DATASET_VERSION,
                "git_commit": git_commit,
                "model_name": name,
                "target": TARGET,
                "person": "Person 4",
            })

            pipeline = Pipeline([
                ("preprocessing", preprocessing),
                ("classifier", classifier),
            ])
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision": float(precision_score(y_test, predictions, zero_division=0)),
                "recall": float(recall_score(y_test, predictions, zero_division=0)),
                "f1": float(f1_score(y_test, predictions, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, probabilities)),
            }

            # Log parameters
            params = dict(model_params[name])
            params.update({
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "n_features": len(X_train.columns),
            })
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log trained model artifact
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
            )

            rows.append({
                "model": name,
                **metrics,
            })
            print(f"Finished & logged to MLflow: {name}")

    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("\nMODEL COMPARISON MATRIX")
    print(results.round(4).to_string(index=False))
    print(f"\nSaved comparison matrix to: {OUTPUT_PATH}")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Features used: {len(X_train.columns)}")
    print("Feature columns:", ", ".join(X_train.columns))


if __name__ == "__main__":
    main()