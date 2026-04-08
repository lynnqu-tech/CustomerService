import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessageChunk

from app.schemas.chat import ChatResponseMode, ConversationTurn
from app.schemas.router import IntentLabel, RoutingDecision
from app.services.chat_service import ChatService, ChatServiceError
from app.services.postgres_service import PostgresServiceError
from app.services.response_service import NO_RETRIEVAL_MESSAGE, ResponseService


class _FakeStreamChain:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def astream(self, _: dict) -> AsyncIterator[AIMessageChunk]:
        for chunk in self._chunks:
            yield AIMessageChunk(content=chunk)


def _build_service() -> ChatService:
    query_router = AsyncMock()
    postgres_service = AsyncMock()
    vector_store = AsyncMock()
    cache_service = AsyncMock()
    llm = MagicMock()
    embeddings = MagicMock()
    embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return ChatService(
        query_router=query_router,
        postgres_service=postgres_service,
        vector_store=vector_store,
        cache_service=cache_service,
        response_service=ResponseService(),
        llm=llm,
        embeddings=embeddings,
    )


def test_stream_chat_returns_db_response_for_order_hit() -> None:
    service = _build_service()
    service._query_router.route.return_value = RoutingDecision(
        label=IntentLabel.ORDER_QUERY,
        route_source="rule",
        matched_value="ORD-1",
        reason="matched",
    )
    service._postgres_service.get_order_by_id.return_value = {
        "order_id": "ORD-1",
        "status": "PAID",
        "created_at": "2026-04-07T10:00:00+00:00",
        "total_amount": "99.00",
    }

    response, stream, decision = asyncio.run(service.stream_chat("s1", "order ORD-1"))

    assert response is not None
    assert response.mode == ChatResponseMode.DB
    assert response.source == "postgres"
    assert stream is None
    assert decision.label == IntentLabel.ORDER_QUERY


def test_stream_chat_returns_guardrail_message_when_no_retrieval_results() -> None:
    service = _build_service()
    service._query_router.route.return_value = RoutingDecision(
        label=IntentLabel.GENERAL_FAQ,
        route_source="llm",
        reason="faq",
    )
    service._cache_service.get_response_cache.return_value = None
    service._vector_store.similarity_search.return_value = []

    response, stream, _ = asyncio.run(service.stream_chat("s1", "how do refunds work"))

    assert response is not None
    assert response.content == NO_RETRIEVAL_MESSAGE
    assert response.source == "rag_guardrail"
    assert stream is None


def test_stream_chat_returns_cached_rag_response_without_llm() -> None:
    service = _build_service()
    service._query_router.route.return_value = RoutingDecision(
        label=IntentLabel.GENERAL_FAQ,
        route_source="llm",
        reason="faq",
    )
    service._cache_service.get_response_cache.return_value = "cached answer"

    response, stream, _ = asyncio.run(service.stream_chat("s1", "coupon expiry"))

    assert response is not None
    assert response.content == "cached answer"
    assert response.source == "response_cache"
    assert stream is None
    service._vector_store.similarity_search.assert_not_awaited()


def test_stream_chat_returns_rag_stream_for_general_faq() -> None:
    service = _build_service()
    service._query_router.route.return_value = RoutingDecision(
        label=IntentLabel.GENERAL_FAQ,
        route_source="llm",
        reason="faq",
    )
    service._cache_service.get_response_cache.return_value = None
    service._cache_service.get_conversation_history.return_value = [
        ConversationTurn(
            question="old q",
            answer="old a",
            created_at=datetime(2026, 4, 7, tzinfo=timezone.utc),
        )
    ]
    service._vector_store.similarity_search.return_value = [
        MagicMock(doc_id="doc-1", text="Refund takes 3-5 days.", score=0.1)
    ]
    service._build_rag_chain = MagicMock(return_value=_FakeStreamChain(["hello ", "world"]))

    response, stream, _ = asyncio.run(service.stream_chat("s1", "refund process"))

    assert response is None
    assert stream is not None
    content = asyncio.run(service._response_service.collect_stream(stream))
    assert content == "hello world"
    service._cache_service.set_response_cache.assert_awaited_once_with(
        "refund process",
        "hello world",
    )


def test_stream_chat_raises_on_database_failure() -> None:
    service = _build_service()
    service._query_router.route.return_value = RoutingDecision(
        label=IntentLabel.ORDER_QUERY,
        route_source="rule",
        matched_value="ORD-1",
        reason="matched",
    )
    service._postgres_service.get_order_by_id.side_effect = PostgresServiceError("db down")

    try:
        asyncio.run(service.stream_chat("s1", "order ORD-1"))
    except ChatServiceError:
        pass
    else:
        raise AssertionError("Expected ChatServiceError")
