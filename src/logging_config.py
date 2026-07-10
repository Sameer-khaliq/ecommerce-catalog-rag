import logging
import json
import sys
from datetime import datetime, timezone

from src.config import get_settings

settings = get_settings()
def _json_log_record(record: logging.LogRecord) -> str:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": record.levelname,
        "logger": record.name,
        "message": record.getMessage(),
    }
    if record.exc_info:
        payload["exception"] = logging.Formatter().formatException(record.exc_info)
    extra_fields = getattr(record, "extra_fields", None)
    if extra_fields:
        payload.update(extra_fields)
    return json.dumps(payload)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return _json_log_record(record)


def get_logger(name: str) -> logging.Logger:
    
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(settings.log_level)
        logger.propagate = False
    return logger


def log_with_context(logger: logging.Logger, level: str, message: str, **context) -> None:
    """Structured extra fields ke saath log karta hai, e.g.:
    log_with_context(logger, "info", "retrieval completed", latency_ms=123, category="Electronics")
    """
    getattr(logger, level.lower())(message, extra={"extra_fields": context})