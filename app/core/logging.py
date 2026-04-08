import logging
from logging.config import dictConfig

from app.core.request_context import get_request_id, get_session_id


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        record.session_id = get_session_id() or "-"
        return True


def configure_logging(settings) -> None:
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {
                "request_context": {
                    "()": "app.core.logging.RequestContextFilter",
                }
            },
            "formatters": {
                "standard": {
                    "format": (
                        "%(asctime)s | %(levelname)s | %(name)s | "
                        "request_id=%(request_id)s session_id=%(session_id)s | %(message)s"
                    )
                }
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "filters": ["request_context"],
                }
            },
            "root": {
                "level": settings.app_log_level.upper(),
                "handlers": ["default"],
            },
        }
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
