"""Tests for the FastAPI REST API."""

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestPredictEndpoint:
    def test_predict_valid_input(self):
        response = client.post("/predict", json={
            "ph": 7.0, "Hardness": 150.0, "Solids": 500.0,
            "Chloramines": 2.0, "Sulfate": 100.0, "Conductivity": 400.0,
            "Organic_carbon": 1.0, "Trihalomethanes": 50.0, "Turbidity": 1.0,
        })
        assert response.status_code == 200
        data = response.json()
        assert "potable" in data
        assert "confidence" in data
        assert "prediction_label" in data
        assert isinstance(data["potable"], bool)
        assert 0 <= data["confidence"] <= 1

    def test_predict_invalid_ph(self):
        response = client.post("/predict", json={
            "ph": 15.0, "Hardness": 150.0, "Solids": 500.0,
            "Chloramines": 2.0, "Sulfate": 100.0, "Conductivity": 400.0,
            "Organic_carbon": 1.0, "Trihalomethanes": 50.0, "Turbidity": 1.0,
        })
        assert response.status_code == 422  # validation error

    def test_predict_missing_field(self):
        response = client.post("/predict", json={
            "ph": 7.0, "Hardness": 150.0,
        })
        assert response.status_code == 422


class TestBatchEndpoint:
    def test_batch_predict(self):
        samples = [
            {"ph": 7.0, "Hardness": 150.0, "Solids": 500.0,
             "Chloramines": 2.0, "Sulfate": 100.0, "Conductivity": 400.0,
             "Organic_carbon": 1.0, "Trihalomethanes": 50.0, "Turbidity": 1.0},
            {"ph": 3.0, "Hardness": 300.0, "Solids": 40000.0,
             "Chloramines": 10.0, "Sulfate": 400.0, "Conductivity": 700.0,
             "Organic_carbon": 20.0, "Trihalomethanes": 100.0, "Turbidity": 6.0},
        ]
        response = client.post("/predict/batch", json={"samples": samples})
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        assert data["summary"]["total"] == 2

    def test_batch_empty(self):
        response = client.post("/predict/batch", json={"samples": []})
        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["total"] == 0
