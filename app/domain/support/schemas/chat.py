from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    question: str = Field(description="User question")
    answer: str = Field(description="Assistant answer")
    created_at: datetime = Field(description="UTC timestamp of the conversation turn")


class ChatResponseMode(str, Enum):
    DB = "db"
    RAG = "rag"


class ChatRequest(BaseModel):
    session_id: str = Field(description="Stable chat session identifier")
    question: str = Field(description="User input question")


class ChatResponse(BaseModel):
    mode: ChatResponseMode
    session_id: str
    intent: str
    content: str
    source: str
