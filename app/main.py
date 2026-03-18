from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.config import Settings
from app.routes.chat_completions import router as chat_completions_router
from app.routes.completions import router as completions_router
from app.routes.models import router as models_router
from app.routes.responses import router as responses_router
from app.services.raw_io_logger import RawIOLogger
from app.services.request_context import reset_current_request_id, set_current_request_id
from app.services.responses_client import (
    AntigravityResponsesGateway,
    OpenAIResponsesGateway,
    RoutingResponsesGateway,
)

logger = logging.getLogger("compat_proxy")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    raw_logger = RawIOLogger.from_settings(settings)
    if settings.upstream_mode == "messages":
        upstream_gateway = AntigravityResponsesGateway(settings, raw_logger=raw_logger)
    else:
        upstream_gateway = OpenAIResponsesGateway(settings, raw_logger=raw_logger)
    gateway = RoutingResponsesGateway(
        settings=settings,
        upstream_gateway=upstream_gateway,
        raw_logger=raw_logger,
    )
    app.state.settings = settings
    app.state.raw_io_logger = raw_logger
    app.state.responses_gateway = gateway
    yield
    await gateway.close()


app = FastAPI(
    title="Completions Compatibility Proxy",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(completions_router)
app.include_router(chat_completions_router)
app.include_router(models_router)
app.include_router(responses_router)


@app.middleware("http")
async def log_error_request_body(request: Request, call_next):
    body_bytes = await request.body()
    request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex}"
    token = set_current_request_id(request_id)

    async def receive():
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    request = Request(request.scope, receive)
    raw_logger: RawIOLogger | None = getattr(request.app.state, "raw_io_logger", None)
    if raw_logger is not None:
        raw_logger.log(
            "proxy.request",
            {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query": request.url.query,
                "body": body_bytes.decode("utf-8", errors="replace"),
            },
        )
    try:
        response = await call_next(request)
    finally:
        reset_current_request_id(token)
    if raw_logger is not None:
        raw_logger.log(
            "proxy.response",
            {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
            },
        )
    if response.status_code >= 400:
        logger.error(
            "HTTP %s %s -> %s; request body: %s",
            request.method,
            request.url.path,
            response.status_code,
            body_bytes.decode("utf-8", errors="replace"),
        )
    return response


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
