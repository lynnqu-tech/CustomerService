from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from app.api.router import api_router
from app.core.config import get_settings
from app.core.exceptions import install_exception_handlers
from app.core.http_metrics import MetricsMiddleware
from app.core.logging import configure_logging, get_logger
from app.core.request_context import RequestContextMiddleware


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    logger = get_logger(__name__)
    logger.info(
        "application_startup",
        extra={
            "app_env": settings.app_env,
            "postgres_dsn": settings.postgres_dsn_safe,
            "redis_url": settings.redis_url,
            "milvus_endpoint": settings.milvus_endpoint,
        },
    )
    yield
    logger.info("application_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(MetricsMiddleware)

    app.include_router(api_router, prefix=settings.api_v1_prefix)
    install_exception_handlers(app)
    return app


app = create_app()
