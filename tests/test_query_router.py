import asyncio
from unittest.mock import AsyncMock, MagicMock

from app.core.config import Settings
from app.schemas.router import IntentLabel, QueryClassification
from app.services.query_router import QueryRouter


def _build_settings() -> Settings:
    return Settings(
        OPENAI_API_KEY="test-key",
        OPENAI_CHAT_MODEL="gpt-4o-mini",
        OPENAI_ROUTER_TEMPERATURE=0,
        QUERY_ROUTER_ORDER_REGEX=r"(?i)\b(?:order|订单)[\s#:：-]*([A-Z0-9-]{6,32})\b",
        QUERY_ROUTER_LOGISTICS_REGEX=r"(?i)\b(?:tracking|物流单号|运单号|快递单号)[\s#:：-]*([A-Z0-9-]{6,32})\b",
    )


def test_route_matches_order_rule_without_llm() -> None:
    router = QueryRouter(settings=_build_settings(), llm=MagicMock())

    decision = asyncio.run(router.route("请帮我查询订单 ORD-202404"))

    assert decision.label == IntentLabel.ORDER_QUERY
    assert decision.route_source == "rule"
    assert decision.matched_value == "ORD-202404"


def test_route_matches_logistics_rule_without_llm() -> None:
    router = QueryRouter(settings=_build_settings(), llm=MagicMock())

    decision = asyncio.run(router.route("物流单号 TRACK-998877 现在到哪了"))

    assert decision.label == IntentLabel.LOGISTICS_QUERY
    assert decision.route_source == "rule"
    assert decision.matched_value == "TRACK-998877"


def test_route_matches_refund_keyword_without_llm() -> None:
    router = QueryRouter(settings=_build_settings(), llm=MagicMock())

    decision = asyncio.run(router.route("我想申请退款，流程是什么"))

    assert decision.label == IntentLabel.REFUND_QUERY
    assert decision.route_source == "rule"


def test_route_falls_back_to_llm_for_general_question() -> None:
    chain = AsyncMock()
    chain.ainvoke.return_value = QueryClassification(
        label=IntentLabel.GENERAL_FAQ,
        reason="General platform policy question",
    )
    router = QueryRouter(settings=_build_settings(), llm=MagicMock())
    router._build_llm_chain = MagicMock(return_value=chain)

    decision = asyncio.run(router.route("会员积分什么时候过期"))

    assert decision.label == IntentLabel.GENERAL_FAQ
    assert decision.route_source == "llm"
    assert decision.reason == "General platform policy question"


def test_route_falls_back_to_general_faq_on_invalid_llm_output() -> None:
    chain = AsyncMock()
    chain.ainvoke.return_value = QueryClassification.model_construct(
        label="NOT_A_VALID_LABEL",
        reason="invalid",
    )
    router = QueryRouter(settings=_build_settings(), llm=MagicMock())
    router._build_llm_chain = MagicMock(return_value=chain)

    decision = asyncio.run(router.route("这周客服在线时间"))

    assert decision.label == IntentLabel.GENERAL_FAQ
    assert decision.route_source == "llm"
    assert "classification failure" in decision.reason


def test_route_falls_back_to_general_faq_on_llm_exception() -> None:
    chain = AsyncMock()
    chain.ainvoke.side_effect = RuntimeError("openai timeout")
    router = QueryRouter(settings=_build_settings(), llm=MagicMock())
    router._build_llm_chain = MagicMock(return_value=chain)

    decision = asyncio.run(router.route("优惠券什么时候失效"))

    assert decision.label == IntentLabel.GENERAL_FAQ
    assert decision.route_source == "llm"
    assert "classification failure" in decision.reason
