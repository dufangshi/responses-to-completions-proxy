from __future__ import annotations

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.models.legacy_chat_completions import LegacyChatCompletionRequest
from app.services.responses_client import BaseResponsesGateway, UpstreamAPIError
from app.services.transformers import (
    UnsupportedParameterError,
    build_chat_responses_payload,
    build_legacy_chat_completion_response,
)

router = APIRouter()


@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def create_chat_completion(
    completion_request: LegacyChatCompletionRequest,
    request: Request,
) -> JSONResponse:
    gateway: BaseResponsesGateway = request.app.state.responses_gateway
    settings = request.app.state.settings

    try:
        resolved_model = settings.resolve_model(completion_request.model)
        payload = build_chat_responses_payload(completion_request, resolved_model)
    except UnsupportedParameterError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
            },
        )

    upstream_results = []
    for _ in range(completion_request.n):
        try:
            upstream_results.append(await gateway.create_response(payload))
        except UpstreamAPIError as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.payload)

    response_body = build_legacy_chat_completion_response(
        request=completion_request,
        upstream_results=upstream_results,
        response_model_name=completion_request.model or resolved_model,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=response_body.model_dump())
