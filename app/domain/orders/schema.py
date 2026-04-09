from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class OrderRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    order_id: str
    user_id: str
    status: str
    total_amount: Decimal
    created_at: datetime
