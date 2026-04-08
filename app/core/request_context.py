from contextvars import ContextVar
from typing import Callable
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

request_id_ctx_var: ContextVar[str | None] = ContextVar("request_id", default=None)
session_id_ctx_var: ContextVar[str | None] = ContextVar("session_id", default=None)


def get_request_id() -> str | None:
    return request_id_ctx_var.get()


def get_session_id() -> str | None:
    return session_id_ctx_var.get()


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        session_id = request.headers.get("X-Session-ID")

        request_token = request_id_ctx_var.set(request_id)
        session_token = session_id_ctx_var.set(session_id)

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            if session_id:
                response.headers["X-Session-ID"] = session_id
            return response
        finally:
            request_id_ctx_var.reset(request_token)
            session_id_ctx_var.reset(session_token)
