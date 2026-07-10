"""Smoke tests for the Flask application."""

from __future__ import annotations

import pytest

import app as app_module


class DummyModel:
    def __init__(self, prediction: int) -> None:
        self.prediction = prediction

    def predict(self, _input_array):
        return [self.prediction]


@pytest.fixture()
def client():
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as test_client:
        yield test_client


def valid_payload() -> dict[str, str]:
    return {feature: "1.0" for feature in app_module.FEATURE_NAMES}


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint(client):
    response = client.get("/health")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "healthy"
    assert payload["expected_features"] == 30


def test_benign_prediction(client, monkeypatch):
    monkeypatch.setattr(app_module, "model", DummyModel(prediction=0))

    response = client.post("/predict", data=valid_payload())

    assert response.status_code == 200
    assert b"Benign" in response.data


def test_malignant_prediction(client, monkeypatch):
    monkeypatch.setattr(app_module, "model", DummyModel(prediction=1))

    response = client.post("/predict", data=valid_payload())

    assert response.status_code == 200
    assert b"Malignant" in response.data


def test_missing_feature_is_rejected(client):
    payload = valid_payload()
    payload.pop(app_module.FEATURE_NAMES[0])

    response = client.post("/predict", data=payload)

    assert response.status_code == 400
