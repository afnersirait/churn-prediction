import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict():
    customer_data = {
        "customer_id": "TEST001",
        "tenure": 24,
        "monthly_charges": 79.99,
        "total_charges": 1919.76,
        "contract_type": "One year",
        "payment_method": "Credit card",
        "internet_service": "Fiber optic",
        "online_security": "Yes",
        "tech_support": "Yes",
        "streaming_tv": "Yes",
        "streaming_movies": "No"
    }
    
    response = client.post("/predict", json=customer_data)
    
    if response.status_code == 200:
        data = response.json()
        assert "customer_id" in data
        assert "churn_probability" in data
        assert "will_churn" in data
        assert "risk_level" in data
        assert 0 <= data["churn_probability"] <= 1
