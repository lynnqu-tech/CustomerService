from fastapi import Depends

from app.core.config import get_settings
from app.core.security import verify_api_token
from app.db.session import get_session_factory
from app.services.cache_service import CacheService
from app.services.chat_service import ChatService
from app.services.postgres_service import PostgresService
from app.services.query_router import QueryRouter
from app.services.response_service import ResponseService
from app.services.vector_store import MilvusVectorStore


def get_cache_service() -> CacheService:
    return CacheService.from_settings()


def get_chat_service(_: str = Depends(verify_api_token)) -> ChatService:
    settings = get_settings()
    postgres_service = PostgresService(session_factory=get_session_factory(settings))
    vector_store = MilvusVectorStore(settings=settings)
    cache_service = get_cache_service()
    query_router = QueryRouter(settings=settings)
    response_service = ResponseService()
    return ChatService(
        query_router=query_router,
        postgres_service=postgres_service,
        vector_store=vector_store,
        cache_service=cache_service,
        response_service=response_service,
        settings=settings,
    )
