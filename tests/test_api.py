from fastapi.testclient import TestClient

from logsentinel.api import create_app


class StubPredictor:
    def predict_frame(self, frame):
        assert "message" in frame.columns
        return [
            {
                "is_anomaly": True,
                "anomaly_score": 0.93,
                "root_cause": "network_timeout",
                "root_cause_confidence": 0.82,
                "model_version": "test",
            }
        ]


def test_health_and_predict_endpoints():
    client = TestClient(create_app(predictor=StubPredictor()))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    response = client.post(
        "/predict",
        json={
            "message": "ERROR upstream timeout status=504 latency_ms=2400",
            "latency_ms": 2400,
            "error_count_5m": 3,
            "request_count": 80,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["is_anomaly"] is True
    assert body["root_cause"] == "network_timeout"
    assert body["model_version"] == "test"
