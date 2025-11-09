"""Tests for CAV API endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "EDON CAV Engine" in data["model"]


def test_telemetry():
    """Test telemetry endpoint."""
    response = client.get("/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert "request_count" in data
    assert "avg_latency_ms" in data
    assert "uptime_seconds" in data


def test_cav_single_window():
    """Test single window CAV computation."""
    import numpy as np
    
    payload = {
        "EDA": np.random.normal(0, 0.2, 240).tolist(),
        "TEMP": np.random.normal(32, 0.2, 240).tolist(),
        "BVP": np.random.normal(0, 0.5, 240).tolist(),
        "ACC_x": np.random.normal(0, 0.05, 240).tolist(),
        "ACC_y": np.random.normal(0, 0.05, 240).tolist(),
        "ACC_z": np.random.normal(1, 0.05, 240).tolist(),
        "temp_c": 22.0,
        "humidity": 50.0,
        "aqi": 35,
        "local_hour": 14
    }
    
    response = client.post("/cav", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "cav_raw" in data
    assert "cav_smooth" in data
    assert "state" in data
    assert "parts" in data
    assert data["state"] in ["overload", "balanced", "focus", "restorative"]


def test_cav_batch():
    """Test batch CAV computation."""
    import numpy as np
    
    windows = []
    for _ in range(3):
        windows.append({
            "EDA": np.random.normal(0, 0.2, 240).tolist(),
            "TEMP": np.random.normal(32, 0.2, 240).tolist(),
            "BVP": np.random.normal(0, 0.5, 240).tolist(),
            "ACC_x": np.random.normal(0, 0.05, 240).tolist(),
            "ACC_y": np.random.normal(0, 0.05, 240).tolist(),
            "ACC_z": np.random.normal(1, 0.05, 240).tolist(),
            "temp_c": 22.0,
            "humidity": 50.0,
            "aqi": 35,
            "local_hour": 14
        })
    
    payload = {"windows": windows}
    response = client.post("/oem/cav/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "latency_ms" in data
    assert "server_version" in data
    assert len(data["results"]) == 3




