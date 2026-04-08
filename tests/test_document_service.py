import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.schemas.document import DocumentChunk
from app.services.cache_service import CacheService
from app.services.document_service import DocumentService, DocumentServiceError


def _build_settings() -> Settings:
    return Settings(
        OPENAI_API_KEY="test-key",
        OPENAI_EMBEDDING_MODEL="text-embedding-3-small",
        OPENAI_EMBEDDING_BATCH_SIZE=2,
        DOCUMENT_CHUNK_SIZE=20,
        DOCUMENT_CHUNK_OVERLAP=5,
        DOCUMENT_ALLOWED_EXTENSIONS=".pdf,.txt,.md",
        DOCUMENT_MAX_FILE_SIZE_MB=10,
        DOCUMENT_THREAD_POOL_SIZE=20,
    )


def test_parse_document_reads_text_file(tmp_path: Path) -> None:
    file_path = tmp_path / "faq.txt"
    file_path.write_text("refund policy details", encoding="utf-8")

    service = DocumentService(
        vector_store=AsyncMock(),
        settings=_build_settings(),
    )

    result = asyncio.run(service.parse_document(file_path))

    assert result == "refund policy details"
    asyncio.run(service.close())


def test_split_text_creates_overlapping_chunks() -> None:
    service = DocumentService(
        vector_store=AsyncMock(),
        settings=_build_settings(),
    )

    chunks = service.split_text(
        text="abcdefghijklmnopqrstuvwxyz1234567890",
        source="faq.txt",
        metadata={"category": "refund"},
    )

    assert len(chunks) >= 2
    assert chunks[0].metadata["source"] == "faq.txt"
    assert chunks[0].metadata["category"] == "refund"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[1].metadata["chunk_index"] == 1
    asyncio.run(service.close())


def test_parse_document_reads_pdf_file_with_pypdf_stub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = tmp_path / "faq.pdf"
    file_path.write_bytes(b"%PDF-1.4")

    class _FakePage:
        def extract_text(self) -> str:
            return "pdf refund content"

    class _FakePdfReader:
        def __init__(self, _: str) -> None:
            self.pages = [_FakePage()]

    fake_module = types.SimpleNamespace(PdfReader=_FakePdfReader)
    monkeypatch.setitem(sys.modules, "pypdf", fake_module)

    service = DocumentService(
        vector_store=AsyncMock(),
        settings=_build_settings(),
    )

    result = asyncio.run(service.parse_document(file_path))

    assert result == "pdf refund content"
    asyncio.run(service.close())


def test_embed_chunks_uses_redis_cache_before_openai() -> None:
    cached_embedding = [0.1, 0.2, 0.3]
    cache_service = AsyncMock(spec=CacheService)
    cache_service.get_embedding_cache.return_value = cached_embedding
    embeddings = MagicMock()
    embeddings.aembed_documents = AsyncMock()

    service = DocumentService(
        vector_store=AsyncMock(),
        cache_service=cache_service,
        embeddings=embeddings,
        settings=_build_settings(),
    )

    result = asyncio.run(
        service.embed_chunks(
            [
                DocumentChunk(
                    chunk_id="chunk-1",
                    text="refund policy",
                    metadata={"source": "faq.txt"},
                )
            ]
        )
    )

    assert result == [cached_embedding]
    cache_service.get_embedding_cache.assert_awaited_once_with("refund policy")
    embeddings.aembed_documents.assert_not_awaited()
    asyncio.run(service.close())


def test_embed_chunks_batches_missing_cache_and_persists_embeddings() -> None:
    cache_service = AsyncMock(spec=CacheService)
    cache_service.get_embedding_cache.return_value = None
    embeddings = MagicMock()
    embeddings.aembed_documents = AsyncMock(
        side_effect=[
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6]],
        ]
    )

    service = DocumentService(
        vector_store=AsyncMock(),
        cache_service=cache_service,
        embeddings=embeddings,
        settings=_build_settings(),
    )

    result = asyncio.run(
        service.embed_chunks(
            [
                DocumentChunk(chunk_id="c1", text="one", metadata={}),
                DocumentChunk(chunk_id="c2", text="two", metadata={}),
                DocumentChunk(chunk_id="c3", text="three", metadata={}),
            ]
        )
    )

    assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    assert embeddings.aembed_documents.await_count == 2
    assert cache_service.set_embedding_cache.await_count == 3
    asyncio.run(service.close())


def test_ingest_file_upserts_vector_documents(tmp_path: Path) -> None:
    file_path = tmp_path / "faq.md"
    file_path.write_text("refund details for delayed orders", encoding="utf-8")

    vector_store = AsyncMock()
    vector_store.upsert_documents.return_value = {"upsert_count": 2}
    service = DocumentService(
        vector_store=vector_store,
        settings=_build_settings(),
    )

    service.embed_chunks = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    service.split_text = MagicMock(
        return_value=[
            DocumentChunk(chunk_id="c1", text="refund details", metadata={"source": "faq.md"}),
            DocumentChunk(chunk_id="c2", text="delayed orders", metadata={"source": "faq.md"}),
        ]
    )

    result = asyncio.run(service.ingest_file(file_path, metadata={"category": "refund"}))

    assert result.chunk_count == 2
    assert result.upsert_result == {"upsert_count": 2}
    vector_store.upsert_documents.assert_awaited_once()
    asyncio.run(service.close())


def test_parse_document_rejects_unsupported_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "faq.csv"
    file_path.write_text("unsupported", encoding="utf-8")
    service = DocumentService(
        vector_store=AsyncMock(),
        settings=_build_settings(),
    )

    with pytest.raises(DocumentServiceError):
        asyncio.run(service.parse_document(file_path))

    asyncio.run(service.close())
