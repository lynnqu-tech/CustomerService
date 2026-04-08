import asyncio
from functools import cached_property
from time import perf_counter
from typing import Any

from pymilvus import DataType, MilvusClient

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.schemas.document import VectorDocument, VectorSearchResult

logger = get_logger(__name__)


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""


class MilvusVectorStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @cached_property
    def client(self) -> MilvusClient:
        return MilvusClient(
            uri=self._settings.milvus_uri,
            user=self._settings.milvus_user,
            password=self._settings.milvus_password,
            db_name=self._settings.milvus_database,
        )

    async def initialize(self) -> None:
        started_at = perf_counter()
        try:
            exists = await asyncio.to_thread(
                self.client.has_collection,
                self._settings.milvus_collection,
            )
            if not exists:
                await asyncio.to_thread(self._create_collection)
            await asyncio.to_thread(
                self.client.load_collection,
                self._settings.milvus_collection,
            )
        except Exception as exc:
            duration = perf_counter() - started_at
            logger.exception(
                "milvus_initialize_failed",
                extra={
                    "collection": self._settings.milvus_collection,
                    "duration_seconds": round(duration, 6),
                },
            )
            raise VectorStoreError("Failed to initialize Milvus collection") from exc

        duration = perf_counter() - started_at
        logger.info(
            "milvus_initialize_completed",
            extra={
                "collection": self._settings.milvus_collection,
                "duration_seconds": round(duration, 6),
            },
        )

    def _create_collection(self) -> None:
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="doc_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=128,
            description="Primary document chunk identifier",
        )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=65535,
            description="Chunk text content",
        )
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self._settings.milvus_dimension,
            description="Embedding vector",
        )
        schema.add_field(
            field_name="metadata",
            datatype=DataType.JSON,
            description="JSON metadata payload",
        )

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type=self._settings.milvus_index_type,
            metric_type=self._settings.milvus_metric_type,
            params={"nlist": self._settings.milvus_nlist},
        )

        self.client.create_collection(
            collection_name=self._settings.milvus_collection,
            schema=schema,
            index_params=index_params,
            consistency_level=self._settings.milvus_consistency_level,
        )

    async def upsert_documents(self, documents: list[VectorDocument]) -> dict[str, Any]:
        started_at = perf_counter()
        payload = [document.model_dump(mode="json") for document in documents]
        try:
            result = await asyncio.to_thread(
                self.client.upsert,
                collection_name=self._settings.milvus_collection,
                data=payload,
            )
            await asyncio.to_thread(
                self.client.flush,
                self._settings.milvus_collection,
            )
        except Exception as exc:
            duration = perf_counter() - started_at
            logger.exception(
                "milvus_upsert_failed",
                extra={
                    "collection": self._settings.milvus_collection,
                    "document_count": len(documents),
                    "duration_seconds": round(duration, 6),
                },
            )
            raise VectorStoreError("Failed to upsert documents into Milvus") from exc

        duration = perf_counter() - started_at
        logger.info(
            "milvus_upsert_completed",
            extra={
                "collection": self._settings.milvus_collection,
                "document_count": len(documents),
                "duration_seconds": round(duration, 6),
            },
        )
        return result

    async def similarity_search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter_expression: str = "",
    ) -> list[VectorSearchResult]:
        started_at = perf_counter()
        try:
            raw_results = await asyncio.to_thread(
                self.client.search,
                collection_name=self._settings.milvus_collection,
                data=[query_vector],
                limit=top_k,
                filter=filter_expression,
                output_fields=["doc_id", "text", "metadata"],
                anns_field="vector",
                search_params={
                    "metric_type": self._settings.milvus_metric_type,
                    "params": {"nprobe": self._settings.milvus_search_nprobe},
                },
            )
        except Exception as exc:
            duration = perf_counter() - started_at
            logger.exception(
                "milvus_search_failed",
                extra={
                    "collection": self._settings.milvus_collection,
                    "top_k": top_k,
                    "duration_seconds": round(duration, 6),
                },
            )
            raise VectorStoreError("Failed to search Milvus collection") from exc

        results: list[VectorSearchResult] = []
        for item in raw_results[0] if raw_results else []:
            entity = item.get("entity", {})
            results.append(
                VectorSearchResult(
                    doc_id=entity.get("doc_id", ""),
                    text=entity.get("text", ""),
                    metadata=entity.get("metadata", {}),
                    score=float(item.get("distance", 0.0)),
                )
            )

        duration = perf_counter() - started_at
        logger.info(
            "milvus_search_completed",
            extra={
                "collection": self._settings.milvus_collection,
                "top_k": top_k,
                "result_count": len(results),
                "duration_seconds": round(duration, 6),
            },
        )
        return results

    async def close(self) -> None:
        await asyncio.to_thread(self.client.close)
