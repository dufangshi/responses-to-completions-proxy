from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.services.file_store import resolve_openai_payload_file_ids
from app.services.responses_payload import ResponsesValidationError, normalize_responses_input
from app.services.responses_client import BaseResponsesGateway, UpstreamAPIError

router = APIRouter()


def _normalize_responses_input(payload: dict[str, Any]) -> dict[str, Any]:
    return normalize_responses_input(payload)


def _normalize_payload_model(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    settings = request.app.state.settings
    file_store = request.app.state.file_store
    normalized = dict(payload)

    resolved_model, reasoning_effort = settings.resolve_model_and_reasoning(None)
    normalized["model"] = resolved_model

    if reasoning_effort:
        normalized["reasoning"] = {"effort": reasoning_effort}
    else:
        normalized.pop("reasoning", None)
    normalized.pop("speed", None)
    normalized.pop("service_tier", None)
    normalized.pop("user", None)

    normalized = _normalize_responses_input(normalized)

    instructions = normalized.get("instructions")
    if "codex" in resolved_model.lower():
        if not isinstance(instructions, str) or not instructions.strip():
            normalized["instructions"] = "You are a helpful assistant."
    if "codex" in resolved_model.lower():
        normalized.pop("max_output_tokens", None)

    return resolve_openai_payload_file_ids(normalized, file_store)


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

    try:
        payload = _normalize_payload_model(raw_payload, request)
    except ResponsesValidationError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                    "param": exc.param,
                    "code": exc.code,
                }
            },
        )

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
