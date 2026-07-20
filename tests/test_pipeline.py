"""Tests for the complete training and serving workflow."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from serving.main import create_app
from src.data_preprocessing import load_and_split_data
from src.evaluate import evaluate_model
from src.predict import load_artifact, predict
from src.train import train_model


def test_data_split_is_reproducible_and_stratified() -> None:
    first = load_and_split_data(test_size=0.2, random_state=42)
    second = load_and_split_data(test_size=0.2, random_state=42)

    assert first.X_train.shape == (455, 30)
    assert first.X_test.shape == (114, 30)
    assert (first.X_train == second.X_train).all()
    assert set(first.y_train) == {0, 1}


def test_training_prediction_and_evaluation(tmp_path) -> None:
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    _, artifact = train_model(model_path=model_path)

    split = load_and_split_data()
    result = predict(split.X_test[0].tolist(), artifact)
    metrics = evaluate_model(model_path=model_path, metrics_path=metrics_path)

    assert model_path.exists()
    assert result["class_id"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert metrics["accuracy"] >= 0.9
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == metrics


def test_prediction_rejects_wrong_feature_count(tmp_path) -> None:
    model_path = tmp_path / "model.joblib"
    train_model(model_path=model_path)
    artifact = load_artifact(model_path)

    with pytest.raises(ValueError, match="Expected 30 features"):
        predict([1.0, 2.0], artifact)


def test_api_prediction(tmp_path) -> None:
    model_path = tmp_path / "model.joblib"
    train_model(model_path=model_path)
    split = load_and_split_data()
    client = TestClient(create_app(model_path))

    health = client.get("/health")
    response = client.post("/predict", json={"features": split.X_test[0].tolist()})

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "model_ready": True}
    assert response.status_code == 200
    assert response.json()["class_name"] in ("malignant", "benign")
