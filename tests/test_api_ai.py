import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_get_ai_config(client):
    res = client.get("/api/ai/config")
    assert res.status_code == 200
    data = res.json()
    assert "ollama" in data
    assert "gemini" in data
    assert data["ollama"]["model"] == "gemma4:12b"


def test_update_ai_config(client):
    res = client.post("/api/ai/config", json={
        "mode": "hybrid",
        "ollama": {"model": "gemma4:12b", "endpoint": "http://localhost:11434"},
        "gemini": {"api_key": "test-key-123", "model": "gemini-2.0-flash"}
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "updated"
    assert data["config"]["gemini"]["api_key"] == "test-key-123"


def test_get_ai_health(client):
    res = client.get("/api/ai/health")
    assert res.status_code == 200
    data = res.json()
    assert "ollama" in data
    assert "gemini" in data
