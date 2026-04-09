import json
from datetime import datetime, timezone
from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.metrics import record_cache_hit
from app.schemas.chat import ConversationTurn
from app.utils.hashing import sha256_text

logger = get_logger(__name__)


class CacheService:
    def __init__(self, redis_client: Any | None = None, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._redis_client = redis_client

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "CacheService":
        resolved_settings = settings or get_settings()
        return cls(redis_client=cls._create_redis_client(resolved_settings), settings=resolved_settings)

    @staticmethod
    def _create_redis_client(settings: Settings) -> Any:
        try:
            from redis.asyncio import Redis
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("redis package is required to use CacheService") from exc

        return Redis.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)

    async def get_conversation_history(self, session_id: str) -> list[ConversationTurn]:
        payload = await self._get_json(self._session_key(session_id))
        if not payload:
            return []
        return [ConversationTurn.model_validate(item) for item in payload]

    async def append_conversation_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
    ) -> list[ConversationTurn]:
        history = await self.get_conversation_history(session_id)
        history.append(
            ConversationTurn(
                question=question,
                answer=answer,
                created_at=datetime.now(timezone.utc),
            )
        )
        trimmed_history = history[-self._settings.redis_max_conversation_turns :]
        await self._set_json(
            self._session_key(session_id),
            [turn.model_dump(mode="json") for turn in trimmed_history],
            ttl_seconds=self._settings.redis_session_ttl_seconds,
        )
        return trimmed_history

    async def get_response_cache(self, question: str) -> str | None:
        key = self._response_key(question)
        value = await self._safe_get(key)
        if value is None:
            return None
        record_cache_hit("response_cache")
        return self._normalize_redis_value(value)

    async def set_response_cache(self, question: str, response: str) -> None:
        await self._safe_set(
            self._response_key(question),
            response,
            ttl_seconds=self._settings.redis_response_cache_ttl_seconds,
        )

    async def get_embedding_cache(self, text: str) -> list[float] | None:
        payload = await self._safe_get(self._embedding_key(text))
        if payload is None:
            return None
        try:
            record_cache_hit("embedding_cache")
            return json.loads(self._normalize_redis_value(payload))
        except (TypeError, json.JSONDecodeError):
            logger.exception("cache_deserialize_failed", extra={"cache_type": "embedding_cache"})
            return None

    async def set_embedding_cache(self, text: str, embedding: list[float]) -> None:
        await self._safe_set(
            self._embedding_key(text),
            json.dumps(embedding),
            ttl_seconds=self._settings.redis_embedding_cache_ttl_seconds,
        )

    async def delete_session(self, session_id: str) -> None:
        if self._redis_client is None:
            return
        try:
            await self._redis_client.delete(self._session_key(session_id))
        except Exception:
            logger.exception("cache_delete_failed", extra={"cache_type": "session_cache"})

    async def close(self) -> None:
        if self._redis_client is None:
            return
        close_method = getattr(self._redis_client, "aclose", None) or getattr(
            self._redis_client,
            "close",
            None,
        )
        if close_method is None:
            return
        try:
            result = close_method()
            if hasattr(result, "__await__"):
                await result
        except Exception:
            logger.exception("cache_close_failed")

    async def _get_json(self, key: str) -> Any | None:
        payload = await self._safe_get(key)
        if payload is None:
            return None
        try:
            return json.loads(self._normalize_redis_value(payload))
        except (TypeError, json.JSONDecodeError):
            logger.exception("cache_deserialize_failed", extra={"cache_key": key})
            return None

    async def _set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        await self._safe_set(key, json.dumps(value), ttl_seconds=ttl_seconds)

    async def _safe_get(self, key: str) -> Any | None:
        if self._redis_client is None:
            return None
        try:
            return await self._redis_client.get(key)
        except Exception:
            logger.exception("cache_get_failed", extra={"cache_key": key})
            return None

    async def _safe_set(self, key: str, value: str, ttl_seconds: int) -> None:
        if self._redis_client is None:
            return
        try:
            await self._redis_client.set(key, value, ex=ttl_seconds)
        except Exception:
            logger.exception("cache_set_failed", extra={"cache_key": key})

    @staticmethod
    def _normalize_redis_value(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _session_key(self, session_id: str) -> str:
        return f"session:{session_id}"

    def _response_key(self, question: str) -> str:
        return f"response:{sha256_text(question.strip())}"

    def _embedding_key(self, text: str) -> str:
        return f"embedding:{sha256_text(text)}"
