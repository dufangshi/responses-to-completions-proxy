from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.services.responses_client import BaseResponsesGateway, UpstreamAPIError

router = APIRouter()


def _normalize_payload_model(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    settings = request.app.state.settings
    normalized = dict(payload)
    model = normalized.get("model")
    if isinstance(model, str) and model.strip():
        normalized["model"] = settings.resolve_model(model)
    else:
        normalized["model"] = settings.default_upstream_model
    return normalized


def _sse_passthrough(lines: AsyncIterator[str]) -> AsyncIterator[bytes]:
    async def generator() -> AsyncIterator[bytes]:
        async for line in lines:
            yield f"{line}\n".encode("utf-8")

    return generator()


@router.post("/v1/responses")
@router.post("/responses")
async def create_response(request: Request) -> Response:
    gateway: BaseResponsesGateway = request.app.state.responses_gateway
    try:
        raw_payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "message": "Request body must be valid JSON object.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_json",
                }
            },
        )

    if not isinstance(raw_payload, dict):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "message": "Request body must be a JSON object.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_request_body",
                }
            },
        )

    payload = _normalize_payload_model(raw_payload, request)

    if payload.get("stream") is True:
        try:
            upstream_lines = await gateway.stream_response(payload)
        except UpstreamAPIError as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.payload)

        return StreamingResponse(
            _sse_passthrough(upstream_lines),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        upstream_response = await gateway.create_response(payload)
    except UpstreamAPIError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.payload)

    return JSONResponse(status_code=status.HTTP_200_OK, content=upstream_response)
