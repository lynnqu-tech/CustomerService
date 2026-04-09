from fastapi import APIRouter

from app.presentation.http.v1.chat import router as chat_router
from app.presentation.http.v1.health import router as health_router
from app.presentation.http.v1.metrics import router as metrics_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(metrics_router, tags=["metrics"])
api_router.include_router(chat_router, tags=["chat"])
