from fastapi import APIRouter, Response

from app.core.metrics import CONTENT_TYPE_LATEST, render_metrics

router = APIRouter()


@router.get("/metrics", summary="Prometheus metrics endpoint")
async def metrics() -> Response:
    return Response(content=render_metrics(), media_type=CONTENT_TYPE_LATEST)
