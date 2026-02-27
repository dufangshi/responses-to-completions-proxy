from __future__ import annotations

from fastapi import APIRouter, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.models.legacy_completions import CompletionUsage, LegacyCompletionRequest
from app.services.responses_client import BaseResponsesGateway, UpstreamAPIError
from app.services.streaming_adapter import (
    completion_stream_chunk,
    completion_stream_usage_chunk,
    encode_sse_done,
    encode_sse_json,
    extract_stream_error,
    fallback_stream_identity,
    iter_upstream_sse_events,
    wants_usage_chunk,
)
from app.services.transformers import (
    UnsupportedParameterError,
    build_legacy_completion_response,
    build_responses_payload,
    extract_usage,
    map_finish_reason,
)

router = APIRouter()


@router.post("/v1/completions")
@router.post("/completions")
async def create_completion(
    completion_request: LegacyCompletionRequest,
    request: Request,
) -> Response:
    gateway: BaseResponsesGateway = request.app.state.responses_gateway
    settings = request.app.state.settings

    try:
        resolved_model, reasoning_effort = settings.resolve_model_and_reasoning(
            completion_request.model
        )
        payload = build_responses_payload(
            completion_request,
            resolved_model,
            reasoning_effort,
        )
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

    if completion_request.stream:
        if completion_request.n != 1:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": {
                        "message": "stream=true currently supports n=1 only.",
                        "type": "invalid_request_error",
                        "param": "n",
                        "code": None,
                    }
                },
            )

        try:
            upstream_lines = await gateway.stream_response(payload)
        except UpstreamAPIError as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.payload)

        response_model_name = completion_request.model or resolved_model

        async def stream_generator():
            response_id, created = fallback_stream_identity("cmpl")
            usage = CompletionUsage()
            final_finish_reason = "stop"

            async for event in iter_upstream_sse_events(upstream_lines):
                if event == "[DONE]":
                    break
                if not isinstance(event, dict):
                    continue
                event_type = event.get("type")
                if event_type == "response.created":
                    response_obj = event.get("response", {})
                    if isinstance(response_obj, dict):
                        response_id = str(response_obj.get("id") or response_id)
                        created = int(response_obj.get("created_at") or created)
                    continue

                if event_type == "response.output_text.delta":
                    delta = event.get("delta")
                    if isinstance(delta, str) and delta:
                        yield encode_sse_json(
                            completion_stream_chunk(
                                response_id=response_id,
                                created=created,
                                model=response_model_name,
                                text=delta,
                                finish_reason=None,
                            )
                        )
                    continue

                if event_type == "response.completed":
                    response_obj = event.get("response", {})
                    if isinstance(response_obj, dict):
                        final_finish_reason = map_finish_reason(response_obj)
                        usage = extract_usage(response_obj)
                    yield encode_sse_json(
                        completion_stream_chunk(
                            response_id=response_id,
                            created=created,
                            model=response_model_name,
                            text="",
                            finish_reason=final_finish_reason,
                        )
                    )
                    if wants_usage_chunk(completion_request.stream_options):
                        yield encode_sse_json(
                            completion_stream_usage_chunk(
                                response_id=response_id,
                                created=created,
                                model=response_model_name,
                                usage=usage,
                            )
                        )
                    yield encode_sse_done()
                    return

                error_obj = extract_stream_error(event)
                if error_obj is not None:
                    yield encode_sse_json(
                        {"error": error_obj or {"message": "Upstream streaming failed."}}
                    )
                    yield encode_sse_done()
                    return

            yield encode_sse_json(
                completion_stream_chunk(
                    response_id=response_id,
                    created=created,
                    model=response_model_name,
                    text="",
                    finish_reason=final_finish_reason,
                )
            )
            if wants_usage_chunk(completion_request.stream_options):
                yield encode_sse_json(
                    completion_stream_usage_chunk(
                        response_id=response_id,
                        created=created,
                        model=response_model_name,
                        usage=usage,
                    )
                )
            yield encode_sse_done()

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    upstream_results = []
    for _ in range(completion_request.n):
        try:
            upstream_results.append(await gateway.create_response(payload))
        except UpstreamAPIError as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.payload)

    response_body = build_legacy_completion_response(
        request=completion_request,
        upstream_results=upstream_results,
        response_model_name=completion_request.model or resolved_model,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=response_body.model_dump())
