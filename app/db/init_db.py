from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.base import Base
from app.models import logistics, order  # noqa: F401


async def create_database_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
