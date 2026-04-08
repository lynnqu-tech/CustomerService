from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Logistics(Base):
    __tablename__ = "logistics"

    tracking_no: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    order_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("orders.order_id"),
        nullable=False,
        index=True,
    )
    carrier: Mapped[str] = mapped_column(String(64), nullable=False)
    current_location: Mapped[str] = mapped_column(String(255), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
