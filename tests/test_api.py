from fastapi.testclient import TestClient

import app.main as main
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_answer(monkeypatch):
    # Mock the RAG graph so the test doesn't need live Pinecone/Anthropic keys
    # or network access. This keeps CI fast, free, and deterministic.
    monkeypatch.setattr(
        main, "run_rag_graph", lambda q: f"Mocked answer for: {q}")

    response = client.post("/query", json={"question": "What is LangGraph?"})
    assert response.status_code == 200
    assert "answer" in response.json()
    assert len(response.json()["answer"]) > 0
