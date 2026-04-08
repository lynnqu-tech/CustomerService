from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from app.api.deps import get_chat_service
from app.main import app
from app.schemas.chat import ChatResponse, ChatResponseMode
from app.schemas.router import IntentLabel, RoutingDecision
from app.services.chat_service import ChatServiceError


class _StubChatService:
    def __init__(self, mode: str) -> None:
        self._mode = mode

    async def stream_chat(
        self,
        session_id: str,
        question: str,
    ):
        if self._mode == "db":
            return (
                ChatResponse(
                    mode=ChatResponseMode.DB,
                    session_id=session_id,
                    intent=IntentLabel.ORDER_QUERY.value,
                    content=f"db:{question}",
                    source="postgres",
                ),
                None,
                RoutingDecision(
                    label=IntentLabel.ORDER_QUERY,
                    route_source="rule",
                    matched_value="ORD-1",
                    reason="matched",
                ),
            )
        if self._mode == "rag":
            async def _stream() -> AsyncIterator[str]:
                yield "hello "
                yield "world"

            return (
                None,
                _stream(),
                RoutingDecision(
                    label=IntentLabel.GENERAL_FAQ,
                    route_source="llm",
                    reason="faq",
                ),
            )
        raise ChatServiceError("forced failure")


def test_chat_api_requires_bearer_token() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/chat",
        json={"session_id": "s1", "question": "hello"},
    )

    assert response.status_code == 401


def test_chat_api_returns_json_for_db_hit() -> None:
    app.dependency_overrides[get_chat_service] = lambda: _StubChatService("db")
    client = TestClient(app)

    response = client.post(
        "/api/v1/chat",
        headers={"Authorization": "Bearer change-me"},
        json={"session_id": "s1", "question": "order ORD-1"},
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "db"
    assert response.json()["source"] == "postgres"
    app.dependency_overrides.clear()


def test_chat_api_returns_sse_for_rag() -> None:
    app.dependency_overrides[get_chat_service] = lambda: _StubChatService("rag")
    client = TestClient(app)

    with client.stream(
        "POST",
        "/api/v1/chat",
        headers={"Authorization": "Bearer change-me"},
        json={"session_id": "s1", "question": "refund policy"},
    ) as response:
        body = "".join(chunk.decode() if isinstance(chunk, bytes) else chunk for chunk in response.iter_text())

    assert response.status_code == 200
    assert "data: {\"type\": \"token\", \"content\": \"hello \"}" in body
    assert "data: {\"type\": \"token\", \"content\": \"world\"}" in body
    assert "data: {\"type\": \"done\"}" in body
    assert response.headers["content-type"].startswith("text/event-stream")
    app.dependency_overrides.clear()


def test_chat_api_returns_error_payload_on_chat_failure() -> None:
    app.dependency_overrides[get_chat_service] = lambda: _StubChatService("error")
    client = TestClient(app)

    response = client.post(
        "/api/v1/chat",
        headers={"Authorization": "Bearer change-me"},
        json={"session_id": "s1", "question": "refund policy"},
    )

    assert response.status_code == 500
    assert response.json()["code"] == "CHAT_SERVICE_ERROR"
    app.dependency_overrides.clear()
