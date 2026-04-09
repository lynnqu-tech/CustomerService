from collections.abc import AsyncIterator
from time import perf_counter

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.metrics import observe_llm_response_duration, observe_rag_retrieval_duration
from app.schemas.chat import ChatResponse, ChatResponseMode, ConversationTurn
from app.schemas.router import IntentLabel, RoutingDecision
from app.services.cache_service import CacheService
from app.services.postgres_service import PostgresService, PostgresServiceError
from app.services.query_router import QueryRouter
from app.services.response_service import NO_RETRIEVAL_MESSAGE, ResponseService
from app.services.vector_store import MilvusVectorStore, VectorSearchResult

logger = get_logger(__name__)


class ChatServiceError(Exception):
    """Raised when the chat workflow cannot complete safely."""


class ChatService:
    def __init__(
        self,
        query_router: QueryRouter,
        postgres_service: PostgresService,
        vector_store: MilvusVectorStore,
        cache_service: CacheService,
        response_service: ResponseService | None = None,
        llm: ChatOpenAI | None = None,
        embeddings: OpenAIEmbeddings | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._query_router = query_router
        self._postgres_service = postgres_service
        self._vector_store = vector_store
        self._cache_service = cache_service
        self._response_service = response_service or ResponseService()
        self._llm = llm
        self._embeddings = embeddings

    async def stream_chat(
        self,
        session_id: str,
        question: str,
    ) -> tuple[ChatResponse | None, AsyncIterator[str] | None, RoutingDecision]:
        decision = await self._query_router.route(question)
        logger.info(
            "chat_route_decided",
            extra={
                "route_decision": decision.label.value,
                "route_source": decision.route_source,
            },
        )

        if decision.label == IntentLabel.ORDER_QUERY and decision.matched_value:
            return await self._handle_order_query(session_id, question, decision)

        if decision.label == IntentLabel.LOGISTICS_QUERY and decision.matched_value:
            return await self._handle_logistics_query(session_id, question, decision)

        response, stream = await self._handle_rag_flow(
            session_id=session_id,
            question=question,
            decision=decision,
            db_empty=False,
        )
        return response, stream, decision

    async def _handle_order_query(
        self,
        session_id: str,
        question: str,
        decision: RoutingDecision,
    ) -> tuple[ChatResponse | None, AsyncIterator[str] | None, RoutingDecision]:
        try:
            result = await self._postgres_service.get_order_by_id(decision.matched_value or "")
        except PostgresServiceError as exc:
            raise ChatServiceError("Order query failed due to database error") from exc

        if result is not None:
            response = self._response_service.build_order_response(session_id, result)
            await self._cache_service.append_conversation_turn(
                session_id=session_id,
                question=question,
                answer=response.content,
            )
            return response, None, decision

        response, stream = await self._handle_rag_flow(
            session_id=session_id,
            question=question,
            decision=decision,
            db_empty=True,
        )
        return response, stream, decision

    async def _handle_logistics_query(
        self,
        session_id: str,
        question: str,
        decision: RoutingDecision,
    ) -> tuple[ChatResponse | None, AsyncIterator[str] | None, RoutingDecision]:
        try:
            result = await self._postgres_service.get_logistics_by_tracking_no(
                decision.matched_value or ""
            )
        except PostgresServiceError as exc:
            raise ChatServiceError("Logistics query failed due to database error") from exc

        if result is not None:
            response = self._response_service.build_logistics_response(session_id, result)
            await self._cache_service.append_conversation_turn(
                session_id=session_id,
                question=question,
                answer=response.content,
            )
            return response, None, decision

        response, stream = await self._handle_rag_flow(
            session_id=session_id,
            question=question,
            decision=decision,
            db_empty=True,
        )
        return response, stream, decision

    async def _handle_rag_flow(
        self,
        session_id: str,
        question: str,
        decision: RoutingDecision,
        db_empty: bool,
    ) -> tuple[ChatResponse | None, AsyncIterator[str] | None]:
        cached_response = await self._cache_service.get_response_cache(question)
        if cached_response is not None:
            await self._cache_service.append_conversation_turn(
                session_id=session_id,
                question=question,
                answer=cached_response,
            )
            return (
                ChatResponse(
                    mode=ChatResponseMode.RAG,
                    session_id=session_id,
                    intent=decision.label.value,
                    content=cached_response,
                    source="response_cache",
                ),
                None,
            )

        query_vector = await self._embed_query(question)
        retrieval_started_at = perf_counter()
        documents = await self._vector_store.similarity_search(
            query_vector=query_vector,
            top_k=self._settings.rag_top_k,
        )
        observe_rag_retrieval_duration(
            decision.label.value,
            "hit" if documents else "miss",
            perf_counter() - retrieval_started_at,
        )
        if not documents:
            await self._cache_service.append_conversation_turn(
                session_id=session_id,
                question=question,
                answer=NO_RETRIEVAL_MESSAGE,
            )
            return (
                ChatResponse(
                    mode=ChatResponseMode.RAG,
                    session_id=session_id,
                    intent=decision.label.value,
                    content=NO_RETRIEVAL_MESSAGE,
                    source="rag_guardrail",
                ),
                None,
            )

        history = await self._cache_service.get_conversation_history(session_id)
        stream = self._stream_rag_answer(
            session_id=session_id,
            question=question,
            decision=decision,
            documents=documents,
            history=history,
            db_empty=db_empty,
        )
        return None, stream

    async def _embed_query(self, question: str) -> list[float]:
        try:
            return await self._get_embeddings().aembed_query(question)
        except Exception as exc:
            logger.exception("rag_query_embedding_failed", extra={"question": question})
            raise ChatServiceError("Failed to embed query for retrieval") from exc

    async def _stream_rag_answer(
        self,
        session_id: str,
        question: str,
        decision: RoutingDecision,
        documents: list[VectorSearchResult],
        history: list[ConversationTurn],
        db_empty: bool,
    ) -> AsyncIterator[str]:
        context = self._build_context(documents)
        history_text = self._build_history(history)
        restriction = (
            self._response_service.build_db_empty_fallback_notice(decision.label)
            if db_empty
            else ""
        )
        chain = self._build_rag_chain()
        started_at = perf_counter()
        chunks: list[str] = []

        try:
            async for chunk in chain.astream(
                {
                    "question": question,
                    "context": context,
                    "history": history_text,
                    "restriction": restriction,
                }
            ):
                text = getattr(chunk, "content", chunk)
                if not text:
                    continue
                chunks.append(str(text))
                yield str(text)
        except Exception as exc:
            logger.exception("rag_stream_failed", extra={"intent": decision.label.value})
            raise ChatServiceError("Failed to stream RAG response") from exc
        else:
            final_answer = "".join(chunks)
            await self._cache_service.set_response_cache(question, final_answer)
            await self._cache_service.append_conversation_turn(
                session_id=session_id,
                question=question,
                answer=final_answer,
            )
            duration = perf_counter() - started_at
            observe_llm_response_duration(decision.label.value, duration)
            logger.info(
                "rag_stream_completed",
                extra={
                    "intent": decision.label.value,
                    "duration_seconds": round(duration, 6),
                    "context_docs": len(documents),
                },
            )

    def _build_rag_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an enterprise ecommerce customer service assistant. "
                    "Answer only from the retrieved knowledge context. "
                    "If the context is insufficient, say you cannot confirm the information. "
                    "Do not fabricate order status, logistics status, or refund progress. "
                    "{restriction}",
                ),
                (
                    "human",
                    "Conversation history:\n{history}\n\n"
                    "Retrieved knowledge:\n{context}\n\n"
                    "User question:\n{question}",
                ),
            ]
        )
        return prompt | self._get_llm()

    def _build_context(self, documents: list[VectorSearchResult]) -> str:
        return "\n\n".join(
            f"[doc_id={item.doc_id} score={item.score}] {item.text}" for item in documents
        )

    def _build_history(self, history: list[ConversationTurn]) -> str:
        if not history:
            return "No prior conversation."
        return "\n".join(
            f"User: {turn.question}\nAssistant: {turn.answer}" for turn in history
        )

    def _get_llm(self) -> ChatOpenAI:
        return self._llm or ChatOpenAI(
            model=self._settings.openai_chat_model,
            api_key=self._settings.openai_api_key,
            temperature=self._settings.openai_rag_temperature,
            streaming=True,
        )

    def _get_embeddings(self) -> OpenAIEmbeddings:
        return self._embeddings or OpenAIEmbeddings(
            model=self._settings.openai_embedding_model,
            api_key=self._settings.openai_api_key,
        )
