import asyncio
from unittest.mock import MagicMock, call

import pytest

from app.core.config import Settings
from app.schemas.document import VectorDocument
from app.services.vector_store import MilvusVectorStore, VectorStoreError


def _build_settings() -> Settings:
    return Settings(
        MILVUS_HOST="localhost",
        MILVUS_PORT=19530,
        MILVUS_DATABASE="default",
        MILVUS_COLLECTION="customer_service_docs",
        MILVUS_DIMENSION=1536,
        MILVUS_INDEX_TYPE="IVF_FLAT",
        MILVUS_METRIC_TYPE="COSINE",
        MILVUS_NLIST=128,
        MILVUS_SEARCH_NPROBE=8,
    )


def test_initialize_creates_collection_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.has_collection.return_value = False

    store = MilvusVectorStore(settings=_build_settings())
    monkeypatch.setattr(MilvusVectorStore, "client", client)
    create_collection = MagicMock()
    monkeypatch.setattr(store, "_create_collection", create_collection)

    asyncio.run(store.initialize())

    client.has_collection.assert_called_once_with("customer_service_docs")
    create_collection.assert_called_once()
    client.load_collection.assert_called_once_with("customer_service_docs")


def test_initialize_skips_collection_creation_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.has_collection.return_value = True

    store = MilvusVectorStore(settings=_build_settings())
    monkeypatch.setattr(MilvusVectorStore, "client", client)
    create_collection = MagicMock()
    monkeypatch.setattr(store, "_create_collection", create_collection)

    asyncio.run(store.initialize())

    create_collection.assert_not_called()
    client.load_collection.assert_called_once_with("customer_service_docs")


def test_upsert_documents_flushes_after_write(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.upsert.return_value = {"upsert_count": 1}

    store = MilvusVectorStore(settings=_build_settings())
    monkeypatch.setattr(MilvusVectorStore, "client", client)

    result = asyncio.run(
        store.upsert_documents(
            [
                VectorDocument(
                    doc_id="doc-1",
                    text="refund policy",
                    vector=[0.1, 0.2, 0.3],
                    metadata={"source": "faq.md"},
                )
            ]
        )
    )

    assert result == {"upsert_count": 1}
    assert client.method_calls == [
        call.upsert(
            collection_name="customer_service_docs",
            data=[
                {
                    "doc_id": "doc-1",
                    "text": "refund policy",
                    "vector": [0.1, 0.2, 0.3],
                    "metadata": {"source": "faq.md"},
                }
            ],
        ),
        call.flush("customer_service_docs"),
    ]


def test_similarity_search_returns_structured_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.search.return_value = [
        [
            {
                "distance": 0.12,
                "entity": {
                    "doc_id": "doc-1",
                    "text": "Refunds take 3-5 business days.",
                    "metadata": {"source": "refund.md"},
                },
            }
        ]
    ]

    store = MilvusVectorStore(settings=_build_settings())
    monkeypatch.setattr(MilvusVectorStore, "client", client)

    results = asyncio.run(store.similarity_search(query_vector=[0.1, 0.2, 0.3], top_k=3))

    assert len(results) == 1
    assert results[0].doc_id == "doc-1"
    assert results[0].text == "Refunds take 3-5 business days."
    assert results[0].metadata == {"source": "refund.md"}
    assert results[0].score == 0.12


def test_similarity_search_raises_on_client_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.search.side_effect = RuntimeError("milvus unavailable")

    store = MilvusVectorStore(settings=_build_settings())
    monkeypatch.setattr(MilvusVectorStore, "client", client)

    with pytest.raises(VectorStoreError):
        asyncio.run(store.similarity_search(query_vector=[0.1, 0.2, 0.3]))
