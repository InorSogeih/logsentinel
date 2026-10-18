from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve


def plot_pr_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    pr_auc: float,
    output_path: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    figure, axis = plt.subplots(figsize=(5.2, 4.0), dpi=160)
    axis.plot(recall, precision, color="#2B5D85", linewidth=2)
    axis.fill_between(recall, precision, alpha=0.12, color="#2B5D85")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(f"Anomaly detection PR curve (AP={pr_auc:.3f})")
    axis.set_xlim(0, 1.02)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
    *,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 4.8), dpi=160)
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=25, ha="right")
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "#1f2933",
                fontsize=9,
            )
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def plot_model_comparison(baseline: dict[str, object], main: dict[str, object], output_path: Path) -> None:
    metrics = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    x_positions = np.arange(len(metrics))
    width = 0.36
    figure, axis = plt.subplots(figsize=(6.4, 3.8), dpi=160)
    axis.bar(
        x_positions - width / 2,
        [float(baseline[name]) for name in metrics],
        width,
        label="TF-IDF + Logistic Regression",
        color="#9FC3DE",
    )
    axis.bar(
        x_positions + width / 2,
        [float(main[name]) for name in metrics],
        width,
        label="TF-IDF/SVD + calibrated HGB",
        color="#2B5D85",
    )
    axis.set_xticks(x_positions, [name.upper() for name in metrics])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Anomaly detector comparison")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
