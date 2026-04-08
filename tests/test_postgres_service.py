import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.models.logistics import Logistics
from app.models.order import Order
from app.services.postgres_service import PostgresService, PostgresServiceError


def _build_session_factory(execute_result=None, execute_side_effect=None):
    session = AsyncMock()
    session.execute = AsyncMock(return_value=execute_result, side_effect=execute_side_effect)

    session_context = AsyncMock()
    session_context.__aenter__.return_value = session
    session_context.__aexit__.return_value = False

    factory = MagicMock(return_value=session_context)
    return factory


def test_get_order_by_id_returns_structured_dict() -> None:
    order = Order(
        order_id="ORD-001",
        user_id="U-100",
        status="PAID",
        total_amount=Decimal("88.50"),
        created_at=datetime(2026, 4, 7, 8, 0, tzinfo=timezone.utc),
    )
    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = order
    service = PostgresService(session_factory=_build_session_factory(execute_result))

    result = asyncio.run(service.get_order_by_id("ORD-001"))

    assert result is not None
    assert result["order_id"] == "ORD-001"
    assert result["user_id"] == "U-100"
    assert result["status"] == "PAID"
    assert result["total_amount"] == Decimal("88.50")


def test_get_order_by_id_returns_none_when_not_found() -> None:
    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = None
    service = PostgresService(session_factory=_build_session_factory(execute_result))

    result = asyncio.run(service.get_order_by_id("ORD-404"))

    assert result is None


def test_get_logistics_by_tracking_no_returns_structured_dict() -> None:
    logistics = Logistics(
        tracking_no="TRACK-001",
        order_id="ORD-001",
        carrier="SF Express",
        current_location="Shanghai Hub",
        updated_at=datetime(2026, 4, 7, 10, 0, tzinfo=timezone.utc),
    )
    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = logistics
    service = PostgresService(session_factory=_build_session_factory(execute_result))

    result = asyncio.run(service.get_logistics_by_tracking_no("TRACK-001"))

    assert result is not None
    assert result["tracking_no"] == "TRACK-001"
    assert result["order_id"] == "ORD-001"
    assert result["carrier"] == "SF Express"


def test_get_logistics_by_tracking_no_returns_none_when_not_found() -> None:
    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = None
    service = PostgresService(session_factory=_build_session_factory(execute_result))

    result = asyncio.run(service.get_logistics_by_tracking_no("TRACK-404"))

    assert result is None


def test_get_order_by_id_raises_service_error_on_sqlalchemy_failure() -> None:
    service = PostgresService(
        session_factory=_build_session_factory(
            execute_side_effect=SQLAlchemyError("database unavailable")
        )
    )

    with pytest.raises(PostgresServiceError):
        asyncio.run(service.get_order_by_id("ORD-001"))
