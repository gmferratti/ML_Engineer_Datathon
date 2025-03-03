import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.app import app
import pandas as pd
client = TestClient(app)

@pytest.fixture
def mock_load_mlflow_model():
    with patch("src.api.app.load_mlflow_model") as mock:
        yield mock

@pytest.fixture
def mock_load_prediction_data():
    with patch("src.api.app.load_prediction_data") as mock:
        yield mock

@pytest.fixture
def mock_get_model_version():
    with patch("src.api.app.get_model_version") as mock:
        yield mock

def test_health_check(mock_load_mlflow_model, mock_load_prediction_data, mock_get_model_version):
    mock_load_mlflow_model.return_value = MagicMock()
    mock_load_prediction_data.return_value = {"news_features": pd.DataFrame(), "clients_features": pd.DataFrame()}
    mock_get_model_version.return_value = "1.0.0"

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_status"] == "loaded"
    assert data["model_version"] == "1.0.0"
    assert data["environment"] == "dev"
    assert data["data_loaded"] is True

def test_predict(mock_load_mlflow_model, mock_load_prediction_data):
    mock_model = MagicMock()
    mock_load_mlflow_model.return_value = mock_model
    mock_load_prediction_data.return_value = {
        "news_features": pd.DataFrame({"pageId": ["1"], "score": [0.9]}),
        "clients_features": pd.DataFrame({"userId": ["test_user"]}),
    }

    with patch("src.api.app.predict_for_userId") as mock_predict_for_userId:
        mock_predict_for_userId.return_value = ([{"pageId": "1", "score": 0.9}], False)

        response = client.post("/predict", json={"userId": "test_user", "max_results": 5, "minScore": 0.3})
        assert response.status_code == 200
        data = response.json()
        assert data["userId"] == "test_user"
        assert len(data["recommendations"]) == 1
        assert data["recommendations"][0]["news_id"] == "1"
        assert data["recommendations"][0]["score"] == 0.9
        assert data["cold_start"] is False

def test_model_info(mock_load_mlflow_model, mock_load_prediction_data, mock_get_model_version):
    mock_load_mlflow_model.return_value = MagicMock(metadata={"mlflow.runName": "1.0.0"})
    mock_load_prediction_data.return_value = {"news_features": pd.DataFrame(), "clients_features": pd.DataFrame()}
    mock_get_model_version.return_value = "1.0.0"

    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "1.0.0"
    assert data["environment"] == "dev"
    assert "metadata" in data
    assert "cache" in data
    assert data["cache"]["cache_hit"] is False
    assert data["cache"]["cache_size"] == 0