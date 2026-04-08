import asyncio
import json
from datetime import datetime, timezone

from app.core.config import Settings
from app.schemas.chat import ConversationTurn
from app.services.cache_service import CacheService
from app.utils.hashing import sha256_text


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.expirations: dict[str, int] = {}

    async def get(self, key: str):
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value
        if ex is not None:
            self.expirations[key] = ex

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)
        self.expirations.pop(key, None)

    async def aclose(self) -> None:
        return None


def _build_settings() -> Settings:
    return Settings(
        REDIS_URL="redis://localhost:6379/0",
        REDIS_SESSION_TTL_SECONDS=1800,
        REDIS_RESPONSE_CACHE_TTL_SECONDS=300,
        REDIS_EMBEDDING_CACHE_TTL_SECONDS=86400,
        REDIS_MAX_CONVERSATION_TURNS=5,
    )


def test_append_conversation_turn_trims_history_to_five_turns() -> None:
    redis_client = FakeRedis()
    service = CacheService(redis_client=redis_client, settings=_build_settings())

    for index in range(6):
        history = asyncio.run(
            service.append_conversation_turn(
                session_id="session-1",
                question=f"q{index}",
                answer=f"a{index}",
            )
        )

    assert len(history) == 5
    assert history[0].question == "q1"
    assert history[-1].answer == "a5"
    assert redis_client.expirations["session:session-1"] == 1800


def test_get_conversation_history_returns_structured_models() -> None:
    redis_client = FakeRedis()
    turns = [
        ConversationTurn.model_construct(
            question="refund?",
            answer="3-5 days",
            created_at=datetime(2026, 4, 7, tzinfo=timezone.utc),
        ).model_dump(mode="json")
    ]
    redis_client.store["session:s1"] = json.dumps(turns)
    service = CacheService(redis_client=redis_client, settings=_build_settings())

    history = asyncio.run(service.get_conversation_history("s1"))

    assert len(history) == 1
    assert history[0].question == "refund?"
    assert history[0].answer == "3-5 days"


def test_response_cache_uses_ttl_and_hash_key() -> None:
    redis_client = FakeRedis()
    service = CacheService(redis_client=redis_client, settings=_build_settings())

    asyncio.run(service.set_response_cache("where is my order", "cached response"))
    result = asyncio.run(service.get_response_cache("where is my order"))

    assert result == "cached response"
    expected_key = f"response:{sha256_text('where is my order')}"
    assert redis_client.store[expected_key] == "cached response"
    assert redis_client.expirations[expected_key] == 300


def test_embedding_cache_round_trips_vector_json() -> None:
    redis_client = FakeRedis()
    service = CacheService(redis_client=redis_client, settings=_build_settings())

    asyncio.run(service.set_embedding_cache("refund policy", [0.1, 0.2, 0.3]))
    result = asyncio.run(service.get_embedding_cache("refund policy"))

    assert result == [0.1, 0.2, 0.3]
    expected_key = f"embedding:{sha256_text('refund policy')}"
    assert redis_client.expirations[expected_key] == 86400


def test_delete_session_clears_cached_history() -> None:
    redis_client = FakeRedis()
    redis_client.store["session:dead-session"] = "[]"
    service = CacheService(redis_client=redis_client, settings=_build_settings())

    asyncio.run(service.delete_session("dead-session"))

    assert "session:dead-session" not in redis_client.store
