from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import ORJSONResponse

from app.core.logging import get_logger
from app.schemas.common import ErrorResponse


class AppException(Exception):
    def __init__(self, code: str, message: str, detail: str, status_code: int) -> None:
        self.code = code
        self.message = message
        self.detail = detail
        self.status_code = status_code
        super().__init__(message)


def install_exception_handlers(app: FastAPI) -> None:
    logger = get_logger(__name__)

    @app.exception_handler(AppException)
    async def handle_app_exception(_: Request, exc: AppException) -> ORJSONResponse:
        logger.exception(
            "application_error",
            extra={"error_code": exc.code, "detail": exc.detail},
        )
        payload = ErrorResponse(code=exc.code, message=exc.message, detail=exc.detail)
        return ORJSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _: Request,
        exc: RequestValidationError,
    ) -> ORJSONResponse:
        payload = ErrorResponse(
            code="VALIDATION_ERROR",
            message="Request validation failed",
            detail=str(exc),
        )
        return ORJSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=payload.model_dump(),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> ORJSONResponse:
        logger.exception("unexpected_error", extra={"detail": str(exc)})
        payload = ErrorResponse(
            code="INTERNAL_SERVER_ERROR",
            message="Internal server error",
            detail="Unhandled exception occurred",
        )
        return ORJSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=payload.model_dump(),
        )
