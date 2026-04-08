from typing import Any

from pydantic import BaseModel, Field


class VectorDocument(BaseModel):
    doc_id: str = Field(description="Stable primary key stored in Milvus")
    text: str = Field(description="Chunk text content")
    vector: list[float] = Field(description="Embedding vector")
    metadata: dict[str, Any] = Field(description="Document metadata payload")


class VectorSearchResult(BaseModel):
    doc_id: str
    text: str
    metadata: dict[str, Any]
    score: float


class DocumentChunk(BaseModel):
    chunk_id: str = Field(description="Stable chunk identifier")
    text: str = Field(description="Chunk text content")
    metadata: dict[str, Any] = Field(description="Chunk-level metadata")


class DocumentIngestResult(BaseModel):
    source: str
    chunk_count: int
    upsert_result: dict[str, Any]
