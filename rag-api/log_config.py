"""Logging configuration for the AeroBook AI API.

Writes to:
    logs/app.log    -> INFO and above, rotating
    logs/error.log  -> ERROR and CRITICAL, with tracebacks
    console         -> INFO and above

setup_logging() must be called before anything else configures logging.
See attach_handler() for adding CloudWatch on top without clobbering these.
"""

import logging
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5

NOISY_LOGGERS = ("asyncio", "botocore", "boto3", "urllib3", "httpx", "httpcore", "qdrant_client", "s3transfer")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with rotating file handlers and a console."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()  # avoid duplicates on reload

    app_log = RotatingFileHandler(
        LOG_DIR / "app.log", maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    app_log.setLevel(level)
    app_log.setFormatter(formatter)

    error_log = RotatingFileHandler(
        LOG_DIR / "error.log", maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    error_log.setLevel(logging.ERROR)
    error_log.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)

    root.addHandler(app_log)
    root.addHandler(error_log)
    root.addHandler(console)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    logger.info("Logging initialised, writing to %s", LOG_DIR)


def attach_handler(handler: logging.Handler, level: int = logging.INFO) -> None:
    """Add an extra handler (CloudWatch) to the root logger.

    Use this instead of logging.basicConfig. basicConfig is a no-op once the
    root logger has handlers, so calling it after setup_logging() would silently
    drop the handler and you would think CloudWatch was still receiving logs.
    """
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(handler)
    logger.info("Attached extra log handler: %s", type(handler).__name__)


def register_error_handlers(app) -> None:
    """Add request logging and global exception handling."""

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start = time.perf_counter()

        logger.info("[%s] --> %s %s", request_id, request.method, request.url.path)

        try:
            response = await call_next(request)
        except Exception:
            # Log here, then re-raise so the Exception handler below still runs
            # and produces the JSON body. Swallowing it would bypass that.
            logger.critical(
                "[%s] request failed after %.2fs: %s %s",
                request_id,
                time.perf_counter() - start,
                request.method,
                request.url.path,
                exc_info=True,
            )
            raise

        duration = time.perf_counter() - start
        logger.info(
            "[%s] <-- %s %s | %s | %.2fs",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", "-")
        logger.warning("[%s] validation error on %s: %s", request_id, request.url.path, exc.errors())
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": "Invalid request",
                "detail": exc.errors(),
                "request_id": request_id,
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        request_id = getattr(request.state, "request_id", "-")
        logger.warning("[%s] HTTP %s on %s: %s", request_id, exc.status_code, request.url.path, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"success": False, "error": exc.detail, "request_id": request_id},
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "-")
        logger.critical("[%s] unhandled exception on %s: %s", request_id, request.url.path, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Internal server error", "request_id": request_id},
        )

    logger.info("Error handlers registered")