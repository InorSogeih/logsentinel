import pandas as pd

from logsentinel.features import add_derived_features, normalize_log


def test_normalize_log_masks_volatile_values():
    text = (
        "ERROR host=10.0.0.12 request_id=550e8400-e29b-41d4-a716-446655440000 "
        "path=/var/lib/containerd latency_ms=1832"
    )
    normalized = normalize_log(text)

    assert "<ip>" in normalized
    assert "<uuid>" in normalized
    assert "<path>" in normalized
    assert "<num>" in normalized
    assert "10.0.0.12" not in normalized


def test_derived_features_are_computed_from_window_metrics():
    frame = pd.DataFrame(
        [
            {
                "message": "ERROR timeout | WARN retry",
                "latency_ms": 1000.0,
                "retry_count": 4,
                "restart_count": 2,
                "cpu_usage": 0.5,
                "memory_usage": 0.9,
                "disk_usage": 0.4,
                "error_count_5m": 5,
                "request_count": 100,
            }
        ]
    )
    prepared = add_derived_features(frame)

    assert prepared.loc[0, "error_ratio"] == 0.05
    assert prepared.loc[0, "retry_rate"] == 0.04
    assert prepared.loc[0, "resource_pressure"] == 0.9
    assert prepared.loc[0, "has_error"] == 1
    assert prepared.loc[0, "log_line_count"] == 2
