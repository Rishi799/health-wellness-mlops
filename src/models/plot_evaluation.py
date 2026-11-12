"""Evaluation charts for the champion model and the 7-model comparison.

Reads the exported champion (models/model.skops) and the held-out test set,
and writes four charts to reports/figures/:

    ml_model_comparison.png   ROC-AUC, recall and F1 of the 7 models
    ml_roc_curve.png          ROC curve of the champion
    ml_confusion_matrix.png   caught / missed / false alarms on the test set
    ml_feature_effects.png    Logistic Regression coefficients (what drives risk)

Run from the repository root (after `dvc pull`):
    python -m src.models.plot_evaluation
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.api.predictor import Predictor

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "reports" / "figures"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
COMPARISON_PATH = ROOT / "reports" / "model_comparison.csv"
TARGET = "cardiometabolic"
NAVY, TEAL, GREY = "#16324F", "#278C87", "#B8C2CC"


def plot_model_comparison(out: Path) -> None:
    results = pd.read_csv(COMPARISON_PATH).set_index("model").sort_values("roc_auc", ascending=False)
    x = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, (metric, colour) in enumerate((("roc_auc", NAVY), ("recall", TEAL), ("f1", GREY))):
        bars = ax.bar(x + (i - 1) * 0.27, results[metric], 0.27, color=colour, label=metric.replace("_", "-").upper())
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, results.index, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_title("Seven models on the same test set (higher is better)")
    ax.legend(frameon=False, ncol=3)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_roc(y: np.ndarray, probability: np.ndarray, out: Path) -> None:
    fpr, tpr, _ = roc_curve(y, probability)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=NAVY, lw=2, label=f"Logistic Regression (AUC = {roc_auc_score(y, probability):.3f})")
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Random guess (AUC = 0.5)")
    ax.set_xlabel("False positive rate (healthy people flagged)")
    ax.set_ylabel("True positive rate = recall")
    ax.set_title("ROC curve — champion model")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_confusion(y: np.ndarray, predicted: np.ndarray, out: Path) -> None:
    matrix = confusion_matrix(y, predicted)
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]}\n{labels[i][j]}", ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else NAVY, fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1], ["Predicted\nnot at risk", "Predicted\nat risk"])
    ax.set_yticks([0, 1], ["Actually\nnot at risk", "Actually\nat risk"])
    ax.set_title("Confusion matrix — test set, threshold 0.5")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_feature_effects(model, out: Path, top: int = 12) -> None:
    names = model.named_steps["preprocessing"].get_feature_names_out()
    coef = pd.Series(model.named_steps["classifier"].coef_[0], index=[n.split("__", 1)[1] for n in names])
    coef = coef.reindex(coef.abs().sort_values(ascending=False).index[:top]).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(coef.index, coef.values, color=[NAVY if v > 0 else TEAL for v in coef.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coefficient on scaled features (right = raises risk)")
    ax.set_title("What drives the prediction")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    predictor = Predictor()
    predictor.load()
    test = pd.read_csv(TEST_PATH)
    y = test[TARGET].to_numpy()
    probability = predictor.model.predict_proba(test[predictor.features])[:, 1]
    threshold = float(predictor.model_info.get("decision_threshold", 0.5))

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_model_comparison(FIGURES / "ml_model_comparison.png")
    plot_roc(y, probability, FIGURES / "ml_roc_curve.png")
    plot_confusion(y, (probability >= threshold).astype(int), FIGURES / "ml_confusion_matrix.png")
    plot_feature_effects(predictor.model, FIGURES / "ml_feature_effects.png")
    print(f"[plot_evaluation] wrote 4 charts to {FIGURES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
