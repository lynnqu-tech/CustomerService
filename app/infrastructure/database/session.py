from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings, get_settings

SessionFactory = async_sessionmaker[AsyncSession]

_engine: AsyncEngine | None = None
_session_factory: SessionFactory | None = None


def _engine_options(settings: Settings) -> dict[str, object]:
    options: dict[str, object] = {
        "echo": False,
        "future": True,
        "pool_pre_ping": True,
    }

    if settings.postgres_dsn.startswith("sqlite"):
        return options

    options.update(
        {
            "pool_size": settings.postgres_pool_size,
            "max_overflow": settings.postgres_max_overflow,
            "pool_timeout": settings.postgres_pool_timeout,
            "pool_recycle": settings.postgres_pool_recycle,
        }
    )
    return options


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    global _engine

    if _engine is None:
        resolved_settings = settings or get_settings()
        _engine = create_async_engine(
            resolved_settings.postgres_dsn,
            **_engine_options(resolved_settings),
        )
    return _engine


def get_session_factory(settings: Settings | None = None) -> SessionFactory:
    global _session_factory

    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(settings),
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
            class_=AsyncSession,
        )
    return _session_factory


async def get_db_session() -> AsyncIterator[AsyncSession]:
    session_factory = get_session_factory()
    async with session_factory() as session:
        yield session


async def dispose_engine() -> None:
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
