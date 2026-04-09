from enum import Enum

from pydantic import BaseModel, Field


class IntentLabel(str, Enum):
    ORDER_QUERY = "ORDER_QUERY"
    LOGISTICS_QUERY = "LOGISTICS_QUERY"
    REFUND_QUERY = "REFUND_QUERY"
    GENERAL_FAQ = "GENERAL_FAQ"


class QueryClassification(BaseModel):
    label: IntentLabel = Field(description="Intent classification label")
    reason: str = Field(description="Short rationale for the selected label")


class RoutingDecision(BaseModel):
    label: IntentLabel
    route_source: str = Field(description="rule or llm")
    matched_value: str | None = Field(default=None)
    reason: str
