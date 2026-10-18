from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE
from .features import build_feature_transformer


def build_baseline_pipeline(*, min_df: int = 2) -> Pipeline:
    """Sparse TF-IDF + numeric features with a linear baseline."""
    return Pipeline(
        steps=[
            ("features", build_feature_transformer(dense=False, min_df=min_df)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_main_pipeline(
    *,
    n_components: int = 64,
    max_iter: int = 160,
    calibration_folds: int = 3,
    min_df: int = 2,
) -> CalibratedClassifierCV:
    """TF-IDF/SVD + metrics, followed by a calibrated gradient-boosted model."""
    base = Pipeline(
        steps=[
            (
                "features",
                build_feature_transformer(
                    dense=True,
                    n_components=n_components,
                    min_df=min_df,
                ),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=max_iter,
                    learning_rate=0.06,
                    max_leaf_nodes=31,
                    l2_regularization=0.2,
                    early_stopping=False,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return CalibratedClassifierCV(
        estimator=base,
        method="sigmoid",
        cv=calibration_folds,
    )


def build_root_cause_pipeline(
    *,
    n_components: int = 64,
    max_iter: int = 160,
    min_df: int = 2,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "features",
                build_feature_transformer(
                    dense=True,
                    n_components=n_components,
                    min_df=min_df,
                ),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=max_iter,
                    learning_rate=0.06,
                    max_leaf_nodes=31,
                    l2_regularization=0.2,
                    early_stopping=False,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
