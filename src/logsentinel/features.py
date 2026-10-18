from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .config import RANDOM_STATE


BASE_NUMERIC_COLUMNS = [
    "latency_ms",
    "retry_count",
    "restart_count",
    "cpu_usage",
    "memory_usage",
    "disk_usage",
    "error_count_5m",
    "request_count",
    "hour",
]
DERIVED_NUMERIC_COLUMNS = [
    "error_ratio",
    "retry_rate",
    "latency_per_request",
    "resource_pressure",
    "has_error",
    "has_warn",
    "log_line_count",
]
NUMERIC_COLUMNS = BASE_NUMERIC_COLUMNS + DERIVED_NUMERIC_COLUMNS


def normalize_log(text: str) -> str:
    """Mask volatile identifiers while preserving log-template semantics."""
    value = str(text).lower()
    value = re.sub(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        " <uuid> ",
        value,
    )
    value = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", " <ip> ", value)
    value = re.sub(r"\b0x[0-9a-f]+\b", " <hex> ", value)
    value = re.sub(r"/[a-z0-9_.\-/]+", " <path> ", value)
    value = re.sub(r"\b\d+(?:\.\d+)?(?:ms|s|mb|gb)?\b", " <num> ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_log_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(normalize_log)


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    request_count = data["request_count"].clip(lower=1).astype(float)
    message = data["message"].fillna("").astype(str).str.lower()

    data["error_ratio"] = data["error_count_5m"].astype(float) / request_count
    data["retry_rate"] = data["retry_count"].astype(float) / request_count
    data["latency_per_request"] = data["latency_ms"].astype(float) / request_count
    data["resource_pressure"] = data[["cpu_usage", "memory_usage", "disk_usage"]].max(axis=1)
    data["has_error"] = message.str.contains("error", regex=False).astype(int)
    data["has_warn"] = message.str.contains("warn", regex=False).astype(int)
    data["log_line_count"] = message.str.count(r"\|") + 1
    return data


def build_feature_transformer(
    *,
    dense: bool,
    n_components: int = 64,
    min_df: int = 2,
) -> Pipeline:
    text_steps: list[tuple[str, object]] = [
        ("normalize", FunctionTransformer(normalize_log_series, validate=False)),
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=3000,
                min_df=min_df,
                sublinear_tf=True,
                strip_accents="unicode",
            ),
        ),
    ]
    if dense:
        text_steps.append(
            ("svd", TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE))
        )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return Pipeline(
        steps=[
            ("derive", FunctionTransformer(add_derived_features, validate=False)),
            (
                "columns",
                ColumnTransformer(
                    transformers=[
                        ("text", Pipeline(text_steps), "message"),
                        ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
                    ],
                    sparse_threshold=0.0 if dense else 0.3,
                    verbose_feature_names_out=False,
                ),
            ),
        ]
    )


def to_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the derived columns required by the transformers."""
    missing = [column for column in BASE_NUMERIC_COLUMNS + ["message"] if column not in frame]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data = add_derived_features(frame)
    return data.fillna(
        {
            "error_ratio": 0.0,
            "retry_rate": 0.0,
            "latency_per_request": 0.0,
            "resource_pressure": 0.0,
        }
    )
