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

    requested_model = normalized.get("model")
    if not isinstance(requested_model, str):
        requested_model = None
    resolved_model, reasoning_effort = settings.resolve_model_and_reasoning(requested_model)
    normalized["model"] = resolved_model

    if reasoning_effort:
        raw_reasoning = normalized.get("reasoning")
        if isinstance(raw_reasoning, dict):
            merged_reasoning = dict(raw_reasoning)
            merged_reasoning["effort"] = reasoning_effort
            normalized["reasoning"] = merged_reasoning
        else:
            normalized["reasoning"] = {"effort": reasoning_effort}
    else:
        normalized.pop("reasoning", None)

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
