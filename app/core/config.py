from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="enterprise-rag-customer-service", alias="APP_NAME")
    app_env: str = Field(default="local", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_log_level: str = Field(default="INFO", alias="APP_LOG_LEVEL")
    api_v1_prefix: str = Field(default="/api/v1", alias="APP_API_V1_PREFIX")

    api_token: str = Field(default="change-me", alias="API_TOKEN")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
    )
    openai_embedding_batch_size: int = Field(
        default=32,
        alias="OPENAI_EMBEDDING_BATCH_SIZE",
    )
    openai_router_temperature: float = Field(
        default=0.0,
        alias="OPENAI_ROUTER_TEMPERATURE",
    )
    openai_rag_temperature: float = Field(
        default=0.2,
        alias="OPENAI_RAG_TEMPERATURE",
    )

    postgres_host: str = Field(default="postgres", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="customer_service", alias="POSTGRES_DB")
    postgres_user: str = Field(default="postgres", alias="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", alias="POSTGRES_PASSWORD")
    postgres_pool_size: int = Field(default=10, alias="POSTGRES_POOL_SIZE")
    postgres_max_overflow: int = Field(default=20, alias="POSTGRES_MAX_OVERFLOW")
    postgres_pool_timeout: int = Field(default=30, alias="POSTGRES_POOL_TIMEOUT")
    postgres_pool_recycle: int = Field(default=1800, alias="POSTGRES_POOL_RECYCLE")

    redis_url: str = Field(default="redis://redis:6379/0", alias="REDIS_URL")
    redis_session_ttl_seconds: int = Field(
        default=1800,
        alias="REDIS_SESSION_TTL_SECONDS",
    )
    redis_response_cache_ttl_seconds: int = Field(
        default=300,
        alias="REDIS_RESPONSE_CACHE_TTL_SECONDS",
    )
    redis_embedding_cache_ttl_seconds: int = Field(
        default=86400,
        alias="REDIS_EMBEDDING_CACHE_TTL_SECONDS",
    )
    redis_max_conversation_turns: int = Field(
        default=5,
        alias="REDIS_MAX_CONVERSATION_TURNS",
    )

    milvus_host: str = Field(default="milvus", alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")
    milvus_user: str = Field(default="", alias="MILVUS_USER")
    milvus_password: str = Field(default="", alias="MILVUS_PASSWORD")
    milvus_database: str = Field(default="default", alias="MILVUS_DATABASE")
    milvus_collection: str = Field(
        default="customer_service_docs",
        alias="MILVUS_COLLECTION",
    )
    milvus_dimension: int = Field(default=1536, alias="MILVUS_DIMENSION")
    milvus_consistency_level: str = Field(
        default="Bounded",
        alias="MILVUS_CONSISTENCY_LEVEL",
    )
    milvus_index_type: str = Field(default="IVF_FLAT", alias="MILVUS_INDEX_TYPE")
    milvus_metric_type: str = Field(default="COSINE", alias="MILVUS_METRIC_TYPE")
    milvus_nlist: int = Field(default=1024, alias="MILVUS_NLIST")
    milvus_search_nprobe: int = Field(default=16, alias="MILVUS_SEARCH_NPROBE")

    langsmith_tracing: bool = Field(default=True, alias="LANGSMITH_TRACING")
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(
        default="enterprise-rag-customer-service",
        alias="LANGSMITH_PROJECT",
    )

    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"],
        alias="CORS_ORIGINS",
    )
    document_chunk_size: int = Field(default=800, alias="DOCUMENT_CHUNK_SIZE")
    document_chunk_overlap: int = Field(default=120, alias="DOCUMENT_CHUNK_OVERLAP")
    document_max_file_size_mb: int = Field(default=10, alias="DOCUMENT_MAX_FILE_SIZE_MB")
    document_allowed_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".txt", ".md"],
        alias="DOCUMENT_ALLOWED_EXTENSIONS",
    )
    document_thread_pool_size: int = Field(default=20, alias="DOCUMENT_THREAD_POOL_SIZE")
    query_router_order_regex: str = Field(
        default=r"(?i)\b(?:order|订单)[\s#:：-]*([A-Z0-9-]{6,32})\b",
        alias="QUERY_ROUTER_ORDER_REGEX",
    )
    query_router_logistics_regex: str = Field(
        default=r"(?i)\b(?:tracking|物流单号|运单号|快递单号)[\s#:：-]*([A-Z0-9-]{6,32})\b",
        alias="QUERY_ROUTER_LOGISTICS_REGEX",
    )
    rag_top_k: int = Field(default=4, alias="RAG_TOP_K")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> list[str]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        raise TypeError("CORS_ORIGINS must be a list or comma-separated string")

    @field_validator("document_allowed_extensions", mode="before")
    @classmethod
    def parse_allowed_extensions(cls, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip().lower() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [item.strip().lower() for item in value.split(",") if item.strip()]
        raise TypeError("DOCUMENT_ALLOWED_EXTENSIONS must be a list or comma-separated string")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_dsn_safe(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:***"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def milvus_endpoint(self) -> str:
        return f"{self.milvus_host}:{self.milvus_port}"

    @property
    def milvus_uri(self) -> str:
        return f"http://{self.milvus_endpoint}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
