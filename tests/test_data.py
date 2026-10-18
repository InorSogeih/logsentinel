from logsentinel.config import ROOT_CAUSES
from logsentinel.data import generate_log_windows
from logsentinel.features import to_model_frame


def test_generator_has_expected_schema_and_labels():
    frame = generate_log_windows(n_samples=500, seed=7)

    assert len(frame) == 500
    assert 0 < frame["is_anomaly"].mean() < 1
    assert frame.loc[frame["is_anomaly"] == 0, "root_cause"].eq("normal").all()
    assert set(frame.loc[frame["is_anomaly"] == 1, "root_cause"]) == set(ROOT_CAUSES)
    assert frame["message"].str.len().min() > 0

    prepared = to_model_frame(frame)
    assert "error_ratio" in prepared.columns
    assert prepared["resource_pressure"].between(0, 1).all()
