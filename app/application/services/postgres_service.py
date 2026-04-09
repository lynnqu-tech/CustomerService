from collections.abc import Callable
from time import perf_counter

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.logging import get_logger
from app.core.metrics import observe_postgres_query_duration
from app.models.logistics import Logistics
from app.models.order import Order
from app.schemas.logistics import LogisticsRead
from app.schemas.order import OrderRead

logger = get_logger(__name__)

SessionFactory = async_sessionmaker[AsyncSession]


class PostgresServiceError(Exception):
    """Raised when a database operation fails and should not trigger fallback."""


class PostgresService:
    def __init__(
        self,
        session_factory: SessionFactory,
        time_provider: Callable[[], float] = perf_counter,
    ) -> None:
        self._session_factory = session_factory
        self._time_provider = time_provider

    async def get_order_by_id(self, order_id: str) -> dict | None:
        started_at = self._time_provider()
        try:
            async with self._session_factory() as session:
                statement = select(Order).where(Order.order_id == order_id)
                result = await session.execute(statement)
                order = result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            duration = self._time_provider() - started_at
            observe_postgres_query_duration("order", "error", duration)
            logger.exception(
                "postgres_query_failed",
                extra={
                    "query_type": "order",
                    "order_id": order_id,
                    "duration_seconds": round(duration, 6),
                },
            )
            raise PostgresServiceError("Failed to query orders table") from exc

        duration = self._time_provider() - started_at
        observe_postgres_query_duration("order", "hit" if order is not None else "miss", duration)
        logger.info(
            "postgres_query_completed",
            extra={
                "query_type": "order",
                "order_id": order_id,
                "result_found": order is not None,
                "duration_seconds": round(duration, 6),
            },
        )
        if order is None:
            return None
        return OrderRead.model_validate(order).model_dump()

    async def get_logistics_by_tracking_no(self, tracking_no: str) -> dict | None:
        started_at = self._time_provider()
        try:
            async with self._session_factory() as session:
                statement = select(Logistics).where(Logistics.tracking_no == tracking_no)
                result = await session.execute(statement)
                logistics = result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            duration = self._time_provider() - started_at
            observe_postgres_query_duration("logistics", "error", duration)
            logger.exception(
                "postgres_query_failed",
                extra={
                    "query_type": "logistics",
                    "tracking_no": tracking_no,
                    "duration_seconds": round(duration, 6),
                },
            )
            raise PostgresServiceError("Failed to query logistics table") from exc

        duration = self._time_provider() - started_at
        observe_postgres_query_duration(
            "logistics",
            "hit" if logistics is not None else "miss",
            duration,
        )
        logger.info(
            "postgres_query_completed",
            extra={
                "query_type": "logistics",
                "tracking_no": tracking_no,
                "result_found": logistics is not None,
                "duration_seconds": round(duration, 6),
            },
        )
        if logistics is None:
            return None
        return LogisticsRead.model_validate(logistics).model_dump()
