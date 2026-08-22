"""Local fallback observability for request latency and streamed-token telemetry."""

import json
import logging
from pathlib import Path
from typing import Any


LOG_DIRECTORY = Path(__file__).resolve().parent / "logs"
_configured = False


def configure_local_logging() -> None:
    """Write application logs locally when CloudWatch is unavailable."""
    global _configured
    if _configured:
        return
    LOG_DIRECTORY.mkdir(exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    root = logging.getLogger()
    app_handler = logging.FileHandler(LOG_DIRECTORY / "app.log", encoding="utf-8")
    app_handler.setFormatter(formatter)
    root.addHandler(app_handler)
    root.setLevel(logging.INFO)

    for name, filename in (
        ("rag.latency", "latency.log"),
        ("rag.token_stream", "token_stream.log"),
        ("rag.token_usage", "query_token_usage.log"),
    ):
        dedicated_logger = logging.getLogger(name)
        handler = logging.FileHandler(LOG_DIRECTORY / filename, encoding="utf-8")
        handler.setFormatter(formatter)
        dedicated_logger.addHandler(handler)
        dedicated_logger.setLevel(logging.INFO)
        dedicated_logger.propagate = False
    _configured = True


def _write(logger_name: str, event: str, **fields: Any) -> None:
    """Write one structured JSON event per line for easy filtering/analysis."""
    logging.getLogger(logger_name).info(
        json.dumps({"event": event, **fields}, default=str, separators=(",", ":"))
    )


def log_latency(stage: str, duration_seconds: float, **fields: Any) -> None:
    _write(
        "rag.latency",
        "stage_latency",
        stage=stage,
        duration_ms=round(duration_seconds * 1000, 2),
        **fields,
    )


def log_token_stream(**fields: Any) -> None:
    _write("rag.token_stream", "token_chunk", **fields)


def log_query_token_usage(**fields: Any) -> None:
    _write("rag.token_usage", "query_token_usage", **fields)
