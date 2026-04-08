from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import ORJSONResponse, StreamingResponse

from app.api.deps import get_chat_service
from app.core.logging import get_logger
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService, ChatServiceError
from app.utils.sse import sse_event

router = APIRouter()
logger = get_logger(__name__)


@router.post("/chat", summary="Chat with customer service assistant")
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    try:
        response, stream, decision = await chat_service.stream_chat(
            session_id=request.session_id,
            question=request.question,
        )
    except ChatServiceError as exc:
        return ORJSONResponse(
            status_code=500,
            content={
                "code": "CHAT_SERVICE_ERROR",
                "message": "Chat processing failed",
                "detail": str(exc),
            },
        )

    logger.info(
        "chat_request_completed",
        extra={
            "session_id": request.session_id,
            "route_decision": decision.label.value,
            "route_source": decision.route_source,
        },
    )

    if response is not None:
        return ORJSONResponse(content=response.model_dump(mode="json"))

    async def event_generator() -> AsyncIterator[str]:
        try:
            if stream is None:
                yield sse_event({"type": "error", "content": "Missing stream"})
                return
            async for chunk in stream:
                if chunk:
                    yield sse_event({"type": "token", "content": chunk})
            yield sse_event({"type": "done"})
        except ChatServiceError as exc:
            yield sse_event({"type": "error", "content": str(exc)})
        except Exception as exc:  # pragma: no cover
            logger.exception("chat_stream_failed", extra={"detail": str(exc)})
            yield sse_event({"type": "error", "content": "Internal streaming error"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
