from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/interim/people_2017-2018.csv"
TARGET = "cardiometabolic"
FEATURES = ["age", "sex", "education", "income_ratio"]


def main():
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, ["age", "income_ratio"]),
        ("categorical", categorical_pipeline, ["sex", "education"]),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    with mlflow.start_run(run_name="logistic-regression-baseline"):
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }

        mlflow.log_params({
            "model": "LogisticRegression",
            "test_size": 0.2,
            "random_state": 42,
            "class_weight": "balanced",
            "features": ",".join(FEATURES),
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="baseline_model", skops_trusted_types=["numpy.dtype"])

        print("Test set metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print("\nClassification report:")
        print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()
