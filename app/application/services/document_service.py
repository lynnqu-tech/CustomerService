import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_openai import OpenAIEmbeddings

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.schemas.document import DocumentChunk, DocumentIngestResult, VectorDocument
from app.services.cache_service import CacheService
from app.services.vector_store import MilvusVectorStore

logger = get_logger(__name__)


class DocumentServiceError(Exception):
    """Raised when document ingestion fails."""


class DocumentService:
    def __init__(
        self,
        vector_store: MilvusVectorStore,
        cache_service: CacheService | None = None,
        embeddings: OpenAIEmbeddings | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._vector_store = vector_store
        self._cache_service = cache_service
        self._embeddings = embeddings
        self._executor = ThreadPoolExecutor(
            max_workers=self._settings.document_thread_pool_size,
            thread_name_prefix="document-worker",
        )
        self._executor_registered = False
        self._parsers = {
            ".txt": self._read_text_document,
            ".md": self._read_text_document,
            ".pdf": self._read_pdf_document,
        }

    @cached_property
    def embeddings(self) -> OpenAIEmbeddings:
        return self._embeddings or OpenAIEmbeddings(
            model=self._settings.openai_embedding_model,
            api_key=self._settings.openai_api_key,
        )

    async def ingest_file(
        self,
        file_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentIngestResult:
        path = Path(file_path)
        base_metadata = metadata or {}

        text = await self.parse_document(path)
        chunks = self.split_text(
            text=text,
            source=path.name,
            metadata=base_metadata,
        )

        if not chunks:
            raise DocumentServiceError("Document produced no chunks")

        vectors = await self.embed_chunks(chunks)
        documents = [
            VectorDocument(
                doc_id=chunk.chunk_id,
                text=chunk.text,
                vector=vector,
                metadata=chunk.metadata,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        upsert_result = await self._vector_store.upsert_documents(documents)

        logger.info(
            "document_ingested",
            extra={
                "source": path.name,
                "chunk_count": len(chunks),
            },
        )
        return DocumentIngestResult(
            source=path.name,
            chunk_count=len(chunks),
            upsert_result=upsert_result,
        )

    async def parse_document(self, file_path: str | Path) -> str:
        await self._ensure_default_executor()

        path = Path(file_path)
        self._validate_file(path)

        parser = self._parsers.get(path.suffix.lower())
        if parser is None:
            raise DocumentServiceError(f"Unsupported file type: {path.suffix}")

        try:
            return await asyncio.to_thread(parser, path)
        except DocumentServiceError:
            raise
        except Exception as exc:
            logger.exception(
                "document_parse_failed",
                extra={"source": path.name, "suffix": path.suffix.lower()},
            )
            raise DocumentServiceError(f"Failed to parse document: {path.name}") from exc

    def split_text(
        self,
        text: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        normalized_text = " ".join(text.split())
        if not normalized_text:
            return []

        base_metadata = metadata.copy() if metadata else {}
        chunk_size = self._settings.document_chunk_size
        overlap = self._settings.document_chunk_overlap
        if overlap >= chunk_size:
            raise DocumentServiceError("DOCUMENT_CHUNK_OVERLAP must be smaller than DOCUMENT_CHUNK_SIZE")

        chunks: list[DocumentChunk] = []
        start = 0
        index = 0
        step = chunk_size - overlap

        while start < len(normalized_text):
            end = min(start + chunk_size, len(normalized_text))
            chunk_text = normalized_text[start:end].strip()
            if chunk_text:
                chunk_id = f"{source}-{index}-{uuid4().hex[:8]}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        metadata={
                            **base_metadata,
                            "source": source,
                            "chunk_index": index,
                        },
                    )
                )
            index += 1
            start += step

        return chunks

    async def embed_chunks(self, chunks: list[DocumentChunk]) -> list[list[float]]:
        cached_vectors: list[list[float] | None] = [None] * len(chunks)
        missing_texts: list[str] = []
        missing_indices: list[int] = []

        for index, chunk in enumerate(chunks):
            cached_vector = await self._get_cached_embedding(chunk.text)
            if cached_vector is not None:
                cached_vectors[index] = cached_vector
                continue
            missing_indices.append(index)
            missing_texts.append(chunk.text)

        if missing_texts:
            for start in range(0, len(missing_texts), self._settings.openai_embedding_batch_size):
                batch_texts = missing_texts[start : start + self._settings.openai_embedding_batch_size]
                batch_indices = missing_indices[start : start + self._settings.openai_embedding_batch_size]

                try:
                    embedded_batch = await self.embeddings.aembed_documents(batch_texts)
                except Exception as exc:
                    logger.exception(
                        "document_embedding_failed",
                        extra={"batch_size": len(batch_texts)},
                    )
                    raise DocumentServiceError("Failed to create embeddings") from exc

                for original_index, vector in zip(batch_indices, embedded_batch, strict=True):
                    cached_vectors[original_index] = vector
                    await self._set_cached_embedding(chunks[original_index].text, vector)

        return [vector for vector in cached_vectors if vector is not None]

    async def _get_cached_embedding(self, text: str) -> list[float] | None:
        if self._cache_service is None:
            return None
        return await self._cache_service.get_embedding_cache(text)

    async def _set_cached_embedding(self, text: str, embedding: list[float]) -> None:
        if self._cache_service is None:
            return
        await self._cache_service.set_embedding_cache(text, embedding)

    def _validate_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            raise DocumentServiceError(f"Document not found: {path}")
        if path.suffix.lower() not in self._settings.document_allowed_extensions:
            raise DocumentServiceError(f"Unsupported file type: {path.suffix}")
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self._settings.document_max_file_size_mb:
            raise DocumentServiceError(
                f"File size exceeds {self._settings.document_max_file_size_mb}MB limit"
            )

    def _read_text_document(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def _read_pdf_document(self, path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise DocumentServiceError("pypdf is required to parse PDF files") from exc

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    async def _ensure_default_executor(self) -> None:
        if self._executor_registered:
            return

        loop = asyncio.get_running_loop()
        loop.set_default_executor(self._executor)
        self._executor_registered = True

    async def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)
