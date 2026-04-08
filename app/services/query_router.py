import re
from time import perf_counter

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.schemas.router import IntentLabel, QueryClassification, RoutingDecision

logger = get_logger(__name__)

REFUND_KEYWORDS = (
    "refund",
    "\u9000\u6b3e",
    "\u9000\u8d27",
    "\u9000\u94b1",
    "return",
    "chargeback",
)
ORDER_KEYWORDS = ("order", "\u8ba2\u5355")
LOGISTICS_KEYWORDS = (
    "tracking",
    "\u7269\u6d41",
    "\u7269\u6d41\u5355\u53f7",
    "\u8fd0\u5355\u53f7",
    "\u5feb\u9012\u5355\u53f7",
)
GENERIC_IDENTIFIER_PATTERN = re.compile(r"([A-Z]{2,10}-[A-Z0-9-]{3,32})")


class QueryRouter:
    def __init__(
        self,
        settings: Settings | None = None,
        llm: ChatOpenAI | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._order_pattern = re.compile(self._settings.query_router_order_regex)
        self._logistics_pattern = re.compile(self._settings.query_router_logistics_regex)
        self._llm = llm

    async def route(self, question: str) -> RoutingDecision:
        started_at = perf_counter()
        stripped_question = question.strip()

        decision = self._route_by_rule(stripped_question)
        if decision is not None:
            duration = perf_counter() - started_at
            logger.info(
                "query_router_rule_matched",
                extra={
                    "route_decision": decision.label.value,
                    "route_source": decision.route_source,
                    "duration_seconds": round(duration, 6),
                },
            )
            return decision

        decision = await self._route_by_llm(stripped_question)
        duration = perf_counter() - started_at
        logger.info(
            "query_router_llm_completed",
            extra={
                "route_decision": decision.label.value,
                "route_source": decision.route_source,
                "duration_seconds": round(duration, 6),
            },
        )
        return decision

    def _route_by_rule(self, question: str) -> RoutingDecision | None:
        order_match = self._order_pattern.search(question)
        if order_match:
            return RoutingDecision(
                label=IntentLabel.ORDER_QUERY,
                route_source="rule",
                matched_value=order_match.group(1),
                reason="Matched order identifier rule",
            )

        generic_identifier_match = GENERIC_IDENTIFIER_PATTERN.search(question.upper())
        lowered_question = question.lower()
        if generic_identifier_match and any(keyword in lowered_question for keyword in ORDER_KEYWORDS):
            return RoutingDecision(
                label=IntentLabel.ORDER_QUERY,
                route_source="rule",
                matched_value=generic_identifier_match.group(1),
                reason="Matched order keyword and identifier heuristic",
            )

        logistics_match = self._logistics_pattern.search(question)
        if logistics_match:
            return RoutingDecision(
                label=IntentLabel.LOGISTICS_QUERY,
                route_source="rule",
                matched_value=logistics_match.group(1),
                reason="Matched logistics tracking rule",
            )

        if generic_identifier_match and any(
            keyword in lowered_question for keyword in LOGISTICS_KEYWORDS
        ):
            return RoutingDecision(
                label=IntentLabel.LOGISTICS_QUERY,
                route_source="rule",
                matched_value=generic_identifier_match.group(1),
                reason="Matched logistics keyword and identifier heuristic",
            )

        if any(keyword in lowered_question for keyword in REFUND_KEYWORDS):
            return RoutingDecision(
                label=IntentLabel.REFUND_QUERY,
                route_source="rule",
                reason="Matched refund keyword rule",
            )

        return None

    async def _route_by_llm(self, question: str) -> RoutingDecision:
        try:
            chain = self._build_llm_chain()
            result = await chain.ainvoke({"question": question})
            label = result.label if isinstance(result.label, IntentLabel) else result.label
            if label not in {intent.value for intent in IntentLabel} and not isinstance(
                label,
                IntentLabel,
            ):
                raise ValueError("Invalid label returned by LLM")
            resolved_label = label if isinstance(label, IntentLabel) else IntentLabel(label)
            return RoutingDecision(
                label=resolved_label,
                route_source="llm",
                reason=result.reason,
            )
        except Exception as exc:
            logger.exception("query_router_llm_failed", extra={"question": question})
            return RoutingDecision(
                label=IntentLabel.GENERAL_FAQ,
                route_source="llm",
                reason=f"Fallback to GENERAL_FAQ due to classification failure: {exc}",
            )

    def _build_llm_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a query router for an ecommerce customer service assistant. "
                    "Classify the user question into exactly one label: "
                    "ORDER_QUERY, LOGISTICS_QUERY, REFUND_QUERY, GENERAL_FAQ. "
                    "Return only the function calling payload that matches the schema.",
                ),
                ("human", "{question}"),
            ]
        )

        return prompt | self._get_llm().with_structured_output(
            QueryClassification,
            method="function_calling",
            strict=True,
        )

    def _get_llm(self) -> ChatOpenAI:
        return self._llm or ChatOpenAI(
            model=self._settings.openai_chat_model,
            api_key=self._settings.openai_api_key,
            temperature=self._settings.openai_router_temperature,
        )
