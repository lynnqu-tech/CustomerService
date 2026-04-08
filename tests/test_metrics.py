from fastapi.testclient import TestClient

from app.main import app


def test_metrics_endpoint_exposes_required_metrics() -> None:
    client = TestClient(app)

    response = client.get("/api/v1/metrics")

    assert response.status_code == 200
    body = response.text
    assert "cache_hit_count" in body
    assert "total_requests" in body
    assert "postgres_query_duration_seconds" in body
    assert "rag_retrieval_duration_seconds" in body
    assert "llm_response_duration_seconds" in body


def test_metrics_middleware_counts_requests() -> None:
    client = TestClient(app)

    client.get("/api/v1/health")
    response = client.get("/api/v1/metrics")

    assert response.status_code == 200
    assert 'total_requests_total{method="GET",path="/api/v1/health",status_code="200"}' in response.text
