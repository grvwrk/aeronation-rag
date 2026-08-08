"""AeroBook RAG API entry point.

Responsibilities are deliberately thin here:
  - configure logging before anything else emits a record
  - attach CloudWatch on top of the local file handlers
  - create the app, mount the router, register error handlers
  - expose /health

All endpoint logic lives in routes.py, all RAG logic in services.py and
generate.py. The old inline POST /v1/chat handler has been removed. It is
replaced by the one in routes.py; keeping both registered a duplicate path
where the first one registered (this file) always won and the router version
was unreachable dead code.
"""

import atexit
import logging
import os
import socket
import time

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Logging must be configured before importing anything that logs at import time.
from log_config import setup_logging, attach_handler, register_error_handlers

setup_logging()

logger = logging.getLogger(__name__)
start_time = time.perf_counter()

import services  # noqa: E402  (import order is intentional)
from routes import router  # noqa: E402


def setup_cloudwatch() -> None:
    """Attach a CloudWatch handler alongside the local file handlers.

    Non-fatal by design. A logging backend being unreachable should not stop
    the API from serving traffic, and it must not stop local development.
    """
    try:
        import boto3
        import watchtower

        config, secret = services.get_runtime()

        cloudwatch_client = boto3.client(
            "logs",
            aws_access_key_id=secret["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=secret["AWS_SECRET_ACCESS_KEY"],
            region_name=config["AWS_REGION"],
        )

        try:
            ec2 = boto3.Session().resource("ec2", region_name=config["AWS_REGION"])
            reservations = ec2.meta.client.describe_instances().get("Reservations", [])
            if not reservations or not reservations[0].get("Instances"):
                raise ValueError("No EC2 instances found")
            stream_name = reservations[0]["Instances"][0]["InstanceId"]
        except Exception as exc:
            logger.warning("Could not determine EC2 instance ID, using hostname: %s", exc)
            stream_name = socket.gethostname()

        handler = watchtower.CloudWatchLogHandler(
            log_group=config["CLOUDWATCH_LOG_GROUP"],
            stream_name=stream_name,
            boto3_client=cloudwatch_client,
            use_queues=False,
        )

        # Not logging.basicConfig: it is a no-op once handlers exist, which
        # would silently drop this handler after setup_logging() ran.
        attach_handler(handler)
        atexit.register(handler.flush)
        logger.info("CloudWatch logging active on stream %s", stream_name)

    except Exception as exc:
        logger.error("CloudWatch logging unavailable, continuing with local logs only: %s", exc, exc_info=True)


if os.getenv("ENABLE_CLOUDWATCH", "").lower() in {"1", "true", "yes"}:
    setup_cloudwatch()
else:
    logger.info("CloudWatch disabled; set ENABLE_CLOUDWATCH=true to enable it")

app = FastAPI(
    title="Aeronation API",
    version="1.0",
    description="API for the Aeronation RAG system",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
register_error_handlers(app)


@app.get("/health", tags=["Health Check"])
async def health_check() -> JSONResponse:
    """Liveness probe. Deliberately does no I/O."""
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "OK"})


logger.info("API ready in %.2f seconds", time.perf_counter() - start_time)


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting uvicorn server")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
        workers=1,  # JOBS is in-process; more than one worker breaks job status
    )
