from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True


def test_metadata_returns_supported_regions() -> None:
    response = client.get("/metadata")
    assert response.status_code == 200
    payload = response.json()
    assert "SILTARA" in payload["supported_regions"]
    assert payload["targets"] == ["pm2_5", "pm10", "so2"]


def test_predict_accepts_demo_schema() -> None:
    response = client.post(
        "/predict",
        json={
            "region": "SILTARA",
            "pm25": 78,
            "pm10": 214,
            "so2": 34,
            "temperature": 35,
            "humidity": 68,
            "wind_speed": 1.8,
            "timestamp": "2026-06-08T10:00:00",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert set(payload["predictions"]) == {"pm2_5", "pm10", "so2"}
    assert payload["risk"]["recommendation"]
