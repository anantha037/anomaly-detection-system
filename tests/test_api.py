import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.api.dependencies import get_app_state
from src.api.main import app
from src.models.autoencoder import LSTMAutoencoder


def override_get_app_state():
    dummy_model = LSTMAutoencoder(n_features=8, latent_dim=32, n_layers=2)
    dummy_model.eval()
    return {
        "model": dummy_model,
        "threshold": 0.5,
        "device": torch.device("cpu"),
        "alert_summary": {"total_alerts": 10, "alert_rate": 0.05},
    }


app.dependency_overrides[get_app_state] = override_get_app_state


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "threshold_loaded" in data


def test_threshold(client):
    response = client.get("/threshold")
    assert response.status_code == 200
    data = response.json()
    assert "threshold" in data
    assert "multiplier" in data
    assert "method" in data


def test_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "model" in data
    assert "f1_score" in data
    assert "precision" in data
    assert "recall" in data


def test_predict_success(client):
    # Make a valid request with 1 window of shape (30, 8)
    dummy_window = np.random.rand(1, 30, 8).tolist()
    response = client.post("/predict", json={"windows": dummy_window})
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "anomaly_count" in data
    assert "total_windows" in data
    assert len(data["results"]) == 1
    assert "is_anomaly" in data["results"][0]
    assert "reconstruction_error" in data["results"][0]


def test_predict_invalid_shape(client):
    # Invalid shape: window length is 20 instead of 30
    invalid_window = np.random.rand(1, 20, 8).tolist()
    response = client.post("/predict", json={"windows": invalid_window})
    assert response.status_code == 422  # Validation error

    # Invalid shape: feature length is 7 instead of 8
    invalid_window_features = np.random.rand(1, 30, 7).tolist()
    response = client.post("/predict", json={"windows": invalid_window_features})
    assert response.status_code == 422  # Validation error
