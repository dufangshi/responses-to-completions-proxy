from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.models.legacy_completions import CompletionUsage
from app.models.legacy_chat_completions import LegacyChatCompletionRequest
from app.services.responses_client import BaseResponsesGateway, UpstreamAPIError
from app.services.streaming_adapter import (
    chat_stream_chunk,
    chat_stream_usage_chunk,
    encode_sse_done,
    encode_sse_json,
    extract_stream_error,
    fallback_stream_identity,
    iter_upstream_sse_events,
    wants_usage_chunk,
)
from app.services.transformers import (
    UnsupportedParameterError,
    build_chat_responses_payload,
    build_legacy_chat_completion_response,
    extract_tool_calls,
    extract_usage,
    map_finish_reason,
)

router = APIRouter()


def _upsert_stream_tool_state(
    tool_states: dict[str, dict[str, Any]],
    tool_order: list[str],
    *,
    item_id: Any = None,
    call_id: Any = None,
    name: Any = None,
    arguments: Any = None,
    arguments_delta: Any = None,
    completed: bool = False,
) -> dict[str, Any]:
    resolved_item_id = item_id.strip() if isinstance(item_id, str) and item_id.strip() else None
    resolved_call_id = call_id.strip() if isinstance(call_id, str) and call_id.strip() else None
    if not resolved_item_id:
        resolved_item_id = resolved_call_id or f"call_{uuid.uuid4().hex}"

    state = tool_states.get(resolved_item_id)
    if state is None:
        state = {
            "item_id": resolved_item_id,
            "index": len(tool_order),
            "id": resolved_call_id or resolved_item_id,
            "name": "",
            "arguments": "",
            "sent_any": False,
            "completed": False,
        }
        tool_states[resolved_item_id] = state
        tool_order.append(resolved_item_id)

    if resolved_call_id:
        state["id"] = resolved_call_id
    if isinstance(name, str) and name.strip():
        state["name"] = name.strip()
    if isinstance(arguments, str):
        state["arguments"] = arguments
    if isinstance(arguments_delta, str) and arguments_delta:
        current = state.get("arguments")
        current_text = current if isinstance(current, str) else ""
        state["arguments"] = f"{current_text}{arguments_delta}"
    if completed:
        state["completed"] = True
    return state


def _build_tool_call_delta_entry(
    tool_state: dict[str, Any],
    arguments_fragment: str | None = None,
) -> dict[str, Any]:
    function_name = tool_state.get("name")
    if not isinstance(function_name, str):
        function_name = ""

    call_id = tool_state.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        call_id = str(tool_state.get("item_id") or f"call_{uuid.uuid4().hex}")

    arguments_text: str
    if arguments_fragment is not None:
        arguments_text = arguments_fragment
    else:
        raw_arguments = tool_state.get("arguments")
        arguments_text = raw_arguments if isinstance(raw_arguments, str) else ""

    return {
        "index": int(tool_state.get("index", 0)),
        "id": call_id,
        "type": "function",
        "function": {
            "name": function_name,
            "arguments": arguments_text,
        },
    }


def _materialize_stream_tool_calls(
    tool_states: dict[str, dict[str, Any]],
    *,
    only_recoverable: bool = False,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    ordered_states = sorted(
        tool_states.values(),
        key=lambda item: int(item.get("index", 0)),
    )
    for state in ordered_states:
        name = state.get("name")
        if not isinstance(name, str) or not name.strip():
            continue

        arguments = state.get("arguments")
        arguments_text = arguments if isinstance(arguments, str) else ""
        if only_recoverable:
            if not arguments_text:
                continue
            try:
                json.loads(arguments_text)
            except json.JSONDecodeError:
                continue

        calls.append(
            {
                "id": state.get("id") or state.get("item_id"),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments_text,
                },
            }
        )
    return calls


@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def create_chat_completion(
    completion_request: LegacyChatCompletionRequest,
    request: Request,
) -> Response:
    gateway: BaseResponsesGateway = request.app.state.responses_gateway
    settings = request.app.state.settings

    try:
        resolved_model, reasoning_effort = settings.resolve_model_and_reasoning(None)
        payload = build_chat_responses_payload(
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

        response_model_name = resolved_model

        async def stream_generator():
            response_id, created = fallback_stream_identity("chatcmpl")
            usage = CompletionUsage()
            role_sent = False
            final_finish_reason = "stop"
            stream_tool_states: dict[str, dict[str, Any]] = {}
            stream_tool_order: list[str] = []

            def ensure_role_chunk() -> bytes | None:
                nonlocal role_sent
                if role_sent:
                    return None
                role_sent = True
                return encode_sse_json(
                    chat_stream_chunk(
                        response_id=response_id,
                        created=created,
                        model=response_model_name,
                        delta={"role": "assistant"},
                        finish_reason=None,
                    )
                )

            def build_tool_delta_chunks(
                tool_state: dict[str, Any],
                arguments_fragment: str | None = None,
            ) -> list[bytes]:
                tool_name = tool_state.get("name")
                if not isinstance(tool_name, str) or not tool_name.strip():
                    return []
                payloads: list[bytes] = []
                role_chunk = ensure_role_chunk()
                if role_chunk is not None:
                    payloads.append(role_chunk)
                payloads.append(
                    encode_sse_json(
                        chat_stream_chunk(
                            response_id=response_id,
                            created=created,
                            model=response_model_name,
                            delta={
                                "tool_calls": [
                                    _build_tool_call_delta_entry(
                                        tool_state,
                                        arguments_fragment=arguments_fragment,
                                    )
                                ]
                            },
                            finish_reason=None,
                        )
                    )
                )
                tool_state["sent_any"] = True
                return payloads

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

                if event_type == "response.output_item.added":
                    item = event.get("item")
                    if isinstance(item, dict) and item.get("type") == "function_call":
                        state = _upsert_stream_tool_state(
                            stream_tool_states,
                            stream_tool_order,
                            item_id=item.get("id"),
                            call_id=item.get("call_id"),
                            name=item.get("name"),
                            arguments=item.get("arguments"),
                        )
                        final_finish_reason = "tool_calls"
                        for chunk in build_tool_delta_chunks(state):
                            yield chunk
                    continue

                if event_type == "response.function_call_arguments.delta":
                    arguments_delta = event.get("delta")
                    if isinstance(arguments_delta, str) and arguments_delta:
                        state = _upsert_stream_tool_state(
                            stream_tool_states,
                            stream_tool_order,
                            item_id=event.get("item_id"),
                            arguments_delta=arguments_delta,
                        )
                        final_finish_reason = "tool_calls"
                        for chunk in build_tool_delta_chunks(
                            state,
                            arguments_fragment=arguments_delta,
                        ):
                            yield chunk
                    continue

                if event_type == "response.function_call_arguments.done":
                    state = _upsert_stream_tool_state(
                        stream_tool_states,
                        stream_tool_order,
                        item_id=event.get("item_id"),
                        arguments=event.get("arguments"),
                        completed=True,
                    )
                    final_finish_reason = "tool_calls"
                    if not bool(state.get("sent_any")):
                        for chunk in build_tool_delta_chunks(state):
                            yield chunk
                    continue

                if event_type == "response.output_item.done":
                    item = event.get("item")
                    if isinstance(item, dict) and item.get("type") == "function_call":
                        state = _upsert_stream_tool_state(
                            stream_tool_states,
                            stream_tool_order,
                            item_id=item.get("id"),
                            call_id=item.get("call_id"),
                            name=item.get("name"),
                            arguments=item.get("arguments"),
                            completed=True,
                        )
                        final_finish_reason = "tool_calls"
                        if not bool(state.get("sent_any")):
                            for chunk in build_tool_delta_chunks(state):
                                yield chunk
                    continue

                if event_type == "response.output_text.delta":
                    delta = event.get("delta")
                    if not isinstance(delta, str) or not delta:
                        continue
                    role_chunk = ensure_role_chunk()
                    if role_chunk is not None:
                        yield role_chunk
                    yield encode_sse_json(
                        chat_stream_chunk(
                            response_id=response_id,
                            created=created,
                            model=response_model_name,
                            delta={"content": delta},
                            finish_reason=None,
                        )
                    )
                    continue

                if event_type == "response.completed":
                    response_obj = event.get("response", {})
                    tool_calls: list[dict] = []
                    if isinstance(response_obj, dict):
                        final_finish_reason = map_finish_reason(response_obj)
                        usage = extract_usage(response_obj)
                        tool_calls = extract_tool_calls(response_obj)
                    if not tool_calls:
                        tool_calls = _materialize_stream_tool_calls(stream_tool_states)
                    if tool_calls:
                        final_finish_reason = "tool_calls"
                    role_chunk = ensure_role_chunk()
                    if role_chunk is not None:
                        yield role_chunk
                    if tool_calls:
                        if not any(bool(state.get("sent_any")) for state in stream_tool_states.values()):
                            tool_calls_delta = []
                            for tool_index, tool_call in enumerate(tool_calls):
                                function_obj = tool_call.get("function")
                                tool_calls_delta.append(
                                    {
                                        "index": tool_index,
                                        "id": tool_call.get("id"),
                                        "type": tool_call.get("type", "function"),
                                        "function": function_obj if isinstance(function_obj, dict) else {},
                                    }
                                )
                            yield encode_sse_json(
                                chat_stream_chunk(
                                    response_id=response_id,
                                    created=created,
                                    model=response_model_name,
                                    delta={"tool_calls": tool_calls_delta},
                                    finish_reason=None,
                                )
                            )
                    yield encode_sse_json(
                        chat_stream_chunk(
                            response_id=response_id,
                            created=created,
                            model=response_model_name,
                            delta={},
                            finish_reason=final_finish_reason,
                        )
                    )
                    if wants_usage_chunk(completion_request.stream_options):
                        yield encode_sse_json(
                            chat_stream_usage_chunk(
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
                    recoverable_tool_calls = _materialize_stream_tool_calls(
                        stream_tool_states,
                        only_recoverable=True,
                    )
                    if recoverable_tool_calls:
                        final_finish_reason = "tool_calls"
                        role_chunk = ensure_role_chunk()
                        if role_chunk is not None:
                            yield role_chunk
                        if not any(bool(state.get("sent_any")) for state in stream_tool_states.values()):
                            tool_calls_delta = []
                            for tool_index, tool_call in enumerate(recoverable_tool_calls):
                                function_obj = tool_call.get("function")
                                tool_calls_delta.append(
                                    {
                                        "index": tool_index,
                                        "id": tool_call.get("id"),
                                        "type": tool_call.get("type", "function"),
                                        "function": function_obj if isinstance(function_obj, dict) else {},
                                    }
                                )
                            yield encode_sse_json(
                                chat_stream_chunk(
                                    response_id=response_id,
                                    created=created,
                                    model=response_model_name,
                                    delta={"tool_calls": tool_calls_delta},
                                    finish_reason=None,
                                )
                            )
                        yield encode_sse_json(
                            chat_stream_chunk(
                                response_id=response_id,
                                created=created,
                                model=response_model_name,
                                delta={},
                                finish_reason=final_finish_reason,
                            )
                        )
                        yield encode_sse_done()
                        return
                    yield encode_sse_json(
                        {"error": error_obj or {"message": "Upstream streaming failed."}}
                    )
                    yield encode_sse_done()
                    return

            trailing_tool_calls = _materialize_stream_tool_calls(stream_tool_states)
            if trailing_tool_calls:
                final_finish_reason = "tool_calls"
                role_chunk = ensure_role_chunk()
                if role_chunk is not None:
                    yield role_chunk
                if not any(bool(state.get("sent_any")) for state in stream_tool_states.values()):
                    tool_calls_delta = []
                    for tool_index, tool_call in enumerate(trailing_tool_calls):
                        function_obj = tool_call.get("function")
                        tool_calls_delta.append(
                            {
                                "index": tool_index,
                                "id": tool_call.get("id"),
                                "type": tool_call.get("type", "function"),
                                "function": function_obj if isinstance(function_obj, dict) else {},
                            }
                        )
                    yield encode_sse_json(
                        chat_stream_chunk(
                            response_id=response_id,
                            created=created,
                            model=response_model_name,
                            delta={"tool_calls": tool_calls_delta},
                            finish_reason=None,
                        )
                    )

            role_chunk = ensure_role_chunk()
            if role_chunk is not None:
                yield role_chunk
            yield encode_sse_json(
                chat_stream_chunk(
                    response_id=response_id,
                    created=created,
                    model=response_model_name,
                    delta={},
                    finish_reason=final_finish_reason,
                )
            )
            if wants_usage_chunk(completion_request.stream_options):
                yield encode_sse_json(
                    chat_stream_usage_chunk(
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

    response_body = build_legacy_chat_completion_response(
        request=completion_request,
        upstream_results=upstream_results,
        response_model_name=resolved_model,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=response_body.model_dump())
