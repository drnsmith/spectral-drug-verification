from fastapi.testclient import TestClient
from app.api import app, RAW_FEATURE_COLUMNS, PROCESSED_FEATURE_COLUMNS

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_metadata_endpoint():
    response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["raw_num_features"] == len(RAW_FEATURE_COLUMNS)
    assert body["processed_num_features"] == len(PROCESSED_FEATURE_COLUMNS)


def test_predict_endpoint_bad_length():
    bad_payload = {"spectrum": [0.0] * 10}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 400
    body = response.json()
    assert "detail" in body