from sklearn.model_selection import train_test_split

from logsentinel.data import generate_log_windows
from logsentinel.model import (
    build_baseline_pipeline,
    build_main_pipeline,
    build_root_cause_pipeline,
)


def test_models_fit_and_return_probabilities():
    frame = generate_log_windows(n_samples=500, seed=11)
    train_frame, test_frame = train_test_split(
        frame,
        test_size=0.25,
        random_state=0,
        stratify=frame["target"],
    )

    baseline = build_baseline_pipeline(min_df=1)
    baseline.fit(train_frame, train_frame["is_anomaly"])
    baseline_scores = baseline.predict_proba(test_frame)
    assert baseline_scores.shape == (len(test_frame), 2)

    main = build_main_pipeline(
        n_components=8,
        max_iter=30,
        calibration_folds=2,
        min_df=1,
    )
    main.fit(train_frame, train_frame["is_anomaly"])
    main_scores = main.predict_proba(test_frame)[:, 1]
    assert len(main_scores) == len(test_frame)
    assert ((main_scores >= 0) & (main_scores <= 1)).all()

    anomaly_train = train_frame[train_frame["is_anomaly"] == 1]
    root_model = build_root_cause_pipeline(n_components=8, max_iter=30, min_df=1)
    root_model.fit(anomaly_train, anomaly_train["root_cause"])
    assert len(root_model.predict(anomaly_train)) == len(anomaly_train)
