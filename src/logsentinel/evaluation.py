from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_threshold(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    *,
    min_recall: float = 0.80,
) -> tuple[float, float]:
    """Pick an F1-maximizing threshold on validation data.

    The threshold is selected on validation data only; the test split is never
    used for model or threshold selection.
    """
    labels = np.asarray(list(y_true), dtype=int)
    scores = np.asarray(list(probabilities), dtype=float)
    best_threshold = 0.5
    best_f1 = -1.0
    fallback_threshold = 0.5
    fallback_f1 = -1.0

    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (scores >= threshold).astype(int)
        if predictions.sum() == 0:
            continue
        score = f1_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        if score > fallback_f1:
            fallback_f1 = float(score)
            fallback_threshold = float(threshold)
        if recall >= min_recall and score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    if best_f1 < 0:
        return fallback_threshold, fallback_f1
    return best_threshold, best_f1


def binary_metrics(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    *,
    threshold: float,
) -> dict[str, object]:
    labels = np.asarray(list(y_true), dtype=int)
    scores = np.asarray(list(probabilities), dtype=float)
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "confusion_matrix": matrix.tolist(),
    }


def top_k_accuracy(
    y_true: Iterable[str],
    probabilities: np.ndarray,
    classes: Iterable[str],
    *,
    k: int = 2,
) -> float:
    labels = np.asarray(list(y_true))
    class_names = np.asarray(list(classes))
    top_k = class_names[np.argsort(probabilities, axis=1)[:, -k:]]
    hits = [label in candidates for label, candidates in zip(labels, top_k)]
    return float(np.mean(hits))


def root_cause_metrics(
    y_true: Iterable[str],
    predictions: Iterable[str],
    probabilities: np.ndarray,
    classes: Iterable[str],
) -> dict[str, object]:
    labels = np.asarray(list(y_true))
    predicted = np.asarray(list(predictions))
    report = classification_report(
        labels,
        predicted,
        output_dict=True,
        zero_division=0,
    )
    return {
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "top2_accuracy": top_k_accuracy(labels, probabilities, classes, k=2),
        "per_class": {
            label: {
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1": float(values["f1-score"]),
                "support": int(values["support"]),
            }
            for label, values in report.items()
            if label not in {"accuracy", "macro avg", "weighted avg"}
        },
    }
