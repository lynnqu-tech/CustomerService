from collections.abc import Callable
from contextlib import contextmanager
from time import perf_counter

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
except ImportError:  # pragma: no cover
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"
    Counter = None  # type: ignore[assignment]
    Histogram = None  # type: ignore[assignment]
    generate_latest = None  # type: ignore[assignment]


class _NoOpMetric:
    def labels(self, **_: str):
        return self

    def inc(self, amount: int = 1) -> None:
        return None

    def observe(self, value: float) -> None:
        return None


def _build_counter(*args, **kwargs):
    return Counter(*args, **kwargs) if Counter is not None else _NoOpMetric()


def _build_histogram(*args, **kwargs):
    return Histogram(*args, **kwargs) if Histogram is not None else _NoOpMetric()


cache_hit_count = _build_counter(
    "cache_hit_count",
    "Number of cache hits grouped by cache type",
    labelnames=("cache_type",),
)

total_requests = _build_counter(
    "total_requests",
    "Total API requests grouped by method, path and status code",
    labelnames=("method", "path", "status_code"),
)

postgres_query_duration_seconds = _build_histogram(
    "postgres_query_duration_seconds",
    "Duration of PostgreSQL queries in seconds",
    labelnames=("query_type", "result"),
)

rag_retrieval_duration_seconds = _build_histogram(
    "rag_retrieval_duration_seconds",
    "Duration of RAG retrieval in seconds",
    labelnames=("intent", "result"),
)

llm_response_duration_seconds = _build_histogram(
    "llm_response_duration_seconds",
    "Duration of LLM streaming/generation in seconds",
    labelnames=("intent",),
)


def record_cache_hit(cache_type: str) -> None:
    cache_hit_count.labels(cache_type=cache_type).inc()


def record_total_request(method: str, path: str, status_code: int) -> None:
    total_requests.labels(
        method=method,
        path=path,
        status_code=str(status_code),
    ).inc()


def observe_postgres_query_duration(query_type: str, result: str, duration_seconds: float) -> None:
    postgres_query_duration_seconds.labels(query_type=query_type, result=result).observe(
        duration_seconds
    )


def observe_rag_retrieval_duration(intent: str, result: str, duration_seconds: float) -> None:
    rag_retrieval_duration_seconds.labels(intent=intent, result=result).observe(duration_seconds)


def observe_llm_response_duration(intent: str, duration_seconds: float) -> None:
    llm_response_duration_seconds.labels(intent=intent).observe(duration_seconds)


@contextmanager
def track_duration(observer: Callable[[float], None]):
    started_at = perf_counter()
    try:
        yield
    finally:
        observer(perf_counter() - started_at)


def render_metrics() -> bytes:
    if generate_latest is None:  # pragma: no cover
        return b""
    return generate_latest()
