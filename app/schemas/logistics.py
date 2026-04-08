from datetime import datetime

from pydantic import BaseModel, ConfigDict


class LogisticsRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    tracking_no: str
    order_id: str
    carrier: str
    current_location: str
    updated_at: datetime
