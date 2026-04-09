from collections.abc import AsyncIterator

from app.core.logging import get_logger
from app.schemas.chat import ChatResponse, ChatResponseMode
from app.schemas.router import IntentLabel

logger = get_logger(__name__)

NO_RETRIEVAL_MESSAGE = (
    "抱歉，我暂时没有检索到可确认的信息。请稍后重试，或联系人工客服获取帮助。"
)


class ResponseService:
    def build_order_response(self, session_id: str, payload: dict) -> ChatResponse:
        content = (
            f"订单 {payload['order_id']} 当前状态为 {payload['status']}，"
            f"下单时间 {payload['created_at']}，订单金额 {payload['total_amount']}。"
        )
        return ChatResponse(
            mode=ChatResponseMode.DB,
            session_id=session_id,
            intent=IntentLabel.ORDER_QUERY.value,
            content=content,
            source="postgres",
        )

    def build_logistics_response(self, session_id: str, payload: dict) -> ChatResponse:
        content = (
            f"物流单号 {payload['tracking_no']} 当前承运商为 {payload['carrier']}，"
            f"最新位置 {payload['current_location']}，更新时间 {payload['updated_at']}。"
        )
        return ChatResponse(
            mode=ChatResponseMode.DB,
            session_id=session_id,
            intent=IntentLabel.LOGISTICS_QUERY.value,
            content=content,
            source="postgres",
        )

    def build_db_empty_fallback_notice(self, intent: IntentLabel) -> str:
        if intent == IntentLabel.ORDER_QUERY:
            return "未查询到对应订单记录。我只能提供通用帮助信息，不能推断具体订单状态。"
        if intent == IntentLabel.LOGISTICS_QUERY:
            return "未查询到对应物流记录。我只能提供通用帮助信息，不能推断具体物流状态。"
        if intent == IntentLabel.REFUND_QUERY:
            return "当前未查询到可确认的退款记录。我只能说明通用退款流程，不能编造具体退款进度。"
        return ""

    async def collect_stream(self, stream: AsyncIterator[str]) -> str:
        parts: list[str] = []
        async for chunk in stream:
            if chunk:
                parts.append(chunk)
        content = "".join(parts)
        logger.info("rag_stream_collected", extra={"content_length": len(content)})
        return content
