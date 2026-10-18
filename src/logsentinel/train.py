from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from .config import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_REPORTS_DIR,
    MODEL_VERSION,
    RANDOM_STATE,
)
from .data import generate_log_windows
from .evaluation import binary_metrics, choose_threshold, root_cause_metrics
from .features import to_model_frame
from .model import build_baseline_pipeline, build_main_pipeline, build_root_cause_pipeline
from .plots import plot_confusion_matrix, plot_model_comparison, plot_pr_curve


def _load_data(data_path: Path, samples: int, seed: int) -> pd.DataFrame:
    if data_path.exists():
        frame = pd.read_csv(data_path)
        required = {"message", "is_anomaly", "root_cause", "target"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{data_path} is missing columns: {sorted(required - set(frame.columns))}")
        return frame

    frame = generate_log_windows(n_samples=samples, seed=seed)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_path, index=False)
    return frame


def _split_data(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_frame, holdout = train_test_split(
        frame,
        test_size=0.40,
        random_state=seed,
        stratify=frame["target"],
    )
    validation_frame, test_frame = train_test_split(
        holdout,
        test_size=0.50,
        random_state=seed,
        stratify=holdout["target"],
    )
    return train_frame, validation_frame, test_frame


def _fit_anomaly_model(model, train_frame: pd.DataFrame, validation_frame: pd.DataFrame):
    model.fit(train_frame, train_frame["is_anomaly"].astype(int))
    validation_scores = model.predict_proba(validation_frame)[:, 1]
    threshold, _ = choose_threshold(validation_frame["is_anomaly"], validation_scores)
    return model, threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LogSentinel models")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--n-components", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=220)
    parser.add_argument("--calibration-folds", type=int, default=3)
    parser.add_argument("--min-df", type=int, default=2)
    args = parser.parse_args()

    frame = _load_data(args.data_path, args.samples, args.seed)
    frame = to_model_frame(frame)
    train_frame, validation_frame, test_frame = _split_data(frame, args.seed)

    baseline = build_baseline_pipeline(min_df=args.min_df)
    baseline, baseline_threshold = _fit_anomaly_model(
        baseline, train_frame, validation_frame
    )
    baseline_test_scores = baseline.predict_proba(test_frame)[:, 1]
    baseline_test = binary_metrics(
        test_frame["is_anomaly"],
        baseline_test_scores,
        threshold=baseline_threshold,
    )

    main_model = build_main_pipeline(
        n_components=args.n_components,
        max_iter=args.max_iter,
        calibration_folds=args.calibration_folds,
        min_df=args.min_df,
    )
    main_model, main_threshold = _fit_anomaly_model(
        main_model, train_frame, validation_frame
    )
    main_test_scores = main_model.predict_proba(test_frame)[:, 1]
    main_test = binary_metrics(
        test_frame["is_anomaly"],
        main_test_scores,
        threshold=main_threshold,
    )

    anomaly_train = train_frame[train_frame["is_anomaly"] == 1]
    anomaly_test = test_frame[test_frame["is_anomaly"] == 1]
    root_model = build_root_cause_pipeline(
        n_components=args.n_components,
        max_iter=args.max_iter,
        min_df=args.min_df,
    )
    root_model.fit(anomaly_train, anomaly_train["root_cause"])
    root_predictions = root_model.predict(anomaly_test)
    root_probabilities = root_model.predict_proba(anomaly_test)
    root_metrics = root_cause_metrics(
        anomaly_test["root_cause"],
        root_predictions,
        root_probabilities,
        root_model.classes_,
    )

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    ).hexdigest()[:16]
    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "rows": int(len(frame)),
        "anomaly_rate": float(frame["is_anomaly"].mean()),
        "split_sizes": {
            "train": int(len(train_frame)),
            "validation": int(len(validation_frame)),
            "test": int(len(test_frame)),
        },
        "threshold": float(main_threshold),
        "data_fingerprint": fingerprint,
        "sklearn_version": sklearn.__version__,
        "baseline": baseline_test,
        "main": main_test,
        "root_cause": root_metrics,
        "root_cause_classes": [str(label) for label in root_model.classes_],
    }

    bundle = {
        "anomaly_model": main_model,
        "root_cause_model": root_model,
        "threshold": main_threshold,
        "metadata": metadata,
    }
    joblib.dump(bundle, args.model_out, compress=3)

    metrics_path = args.reports_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    plot_pr_curve(
        np.asarray(test_frame["is_anomaly"], dtype=int),
        main_test_scores,
        float(main_test["pr_auc"]),
        args.reports_dir / "pr_curve.png",
    )
    plot_confusion_matrix(
        np.asarray(main_test["confusion_matrix"]),
        ["normal", "anomaly"],
        args.reports_dir / "anomaly_confusion_matrix.png",
        title=f"Anomaly confusion matrix @ threshold={main_threshold:.2f}",
    )
    root_labels = [str(label) for label in root_model.classes_]
    root_matrix = np.zeros((len(root_labels), len(root_labels)), dtype=int)
    index = {label: position for position, label in enumerate(root_labels)}
    for expected, predicted in zip(anomaly_test["root_cause"], root_predictions):
        root_matrix[index[str(expected)], index[str(predicted)]] += 1
    plot_confusion_matrix(
        root_matrix,
        root_labels,
        args.reports_dir / "root_cause_confusion_matrix.png",
        title="Root-cause confusion matrix",
    )
    plot_model_comparison(
        baseline_test,
        main_test,
        args.reports_dir / "model_comparison.png",
    )

    print(
        "Baseline  "
        f"F1={baseline_test['f1']:.3f}  PR-AUC={baseline_test['pr_auc']:.3f}  "
        f"threshold={baseline_threshold:.2f}"
    )
    print(
        "Main      "
        f"F1={main_test['f1']:.3f}  PR-AUC={main_test['pr_auc']:.3f}  "
        f"threshold={main_threshold:.2f}"
    )
    print(
        "Root cause "
        f"macro-F1={root_metrics['macro_f1']:.3f}  "
        f"top-2={root_metrics['top2_accuracy']:.3f}"
    )
    print(f"Model: {args.model_out}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
