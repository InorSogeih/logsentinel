from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import DEFAULT_MODEL_PATH
from .features import to_model_frame


class Predictor:
    def __init__(self, bundle: dict[str, Any], model_path: Path | None = None) -> None:
        for key in ("anomaly_model", "root_cause_model", "threshold", "metadata"):
            if key not in bundle:
                raise ValueError(f"Invalid model bundle: missing {key}")
        self.anomaly_model = bundle["anomaly_model"]
        self.root_cause_model = bundle["root_cause_model"]
        self.threshold = float(bundle["threshold"])
        self.metadata = bundle["metadata"]
        self.model_path = model_path

    @classmethod
    def from_path(cls, path: Path | str) -> "Predictor":
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run `python -m logsentinel.train` first."
            )
        return cls(joblib.load(model_path), model_path=model_path)

    @classmethod
    def from_env(cls) -> "Predictor":
        return cls.from_path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

    def predict_frame(self, frame: pd.DataFrame) -> list[dict[str, object]]:
        prepared = to_model_frame(frame)
        anomaly_scores = self.anomaly_model.predict_proba(prepared)[:, 1]
        anomaly_flags = anomaly_scores >= self.threshold
        root_predictions = self.root_cause_model.predict(prepared)
        root_probabilities = self.root_cause_model.predict_proba(prepared)
        root_classes = self.root_cause_model.classes_

        results: list[dict[str, object]] = []
        for index, (score, is_anomaly) in enumerate(zip(anomaly_scores, anomaly_flags)):
            result: dict[str, object] = {
                "is_anomaly": bool(is_anomaly),
                "anomaly_score": float(score),
                "root_cause": None,
                "root_cause_confidence": None,
                "model_version": str(self.metadata.get("model_version", "unknown")),
            }
            if is_anomaly:
                probabilities = root_probabilities[index]
                best_index = int(np.argmax(probabilities))
                result["root_cause"] = str(root_classes[best_index])
                result["root_cause_confidence"] = float(probabilities[best_index])
            results.append(result)
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one LogSentinel prediction")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--message", required=True)
    parser.add_argument("--severity", default="ERROR")
    parser.add_argument("--latency-ms", type=float, default=120.0)
    parser.add_argument("--retry-count", type=int, default=0)
    parser.add_argument("--restart-count", type=int, default=0)
    parser.add_argument("--cpu-usage", type=float, default=0.3)
    parser.add_argument("--memory-usage", type=float, default=0.4)
    parser.add_argument("--disk-usage", type=float, default=0.4)
    parser.add_argument("--error-count-5m", type=int, default=0)
    parser.add_argument("--request-count", type=int, default=100)
    parser.add_argument("--hour", type=int, default=12)
    args = parser.parse_args()

    frame = pd.DataFrame(
        [
            {
                "message": args.message,
                "severity": args.severity,
                "latency_ms": args.latency_ms,
                "retry_count": args.retry_count,
                "restart_count": args.restart_count,
                "cpu_usage": args.cpu_usage,
                "memory_usage": args.memory_usage,
                "disk_usage": args.disk_usage,
                "error_count_5m": args.error_count_5m,
                "request_count": args.request_count,
                "hour": args.hour,
            }
        ]
    )
    result = Predictor.from_path(args.model).predict_frame(frame)[0]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
