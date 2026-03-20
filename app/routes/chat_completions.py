from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.models.legacy_completions import CompletionUsage
from app.models.legacy_chat_completions import LegacyChatCompletionRequest
from app.services.file_store import resolve_openai_payload_file_ids
from app.services.responses_client import BaseResponsesGateway, UpstreamAPIError
from app.services.responses_session_store import ResponsesSessionState, ResponsesSessionStore
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

_NANOBOT_SESSION_HEADER = "x-nanobot-session-key"
_NANOBOT_RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"


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
    file_store = request.app.state.file_store

    try:
        resolved_model, reasoning_effort = settings.resolve_model_and_reasoning(None)
        payload = build_chat_responses_payload(
            completion_request,
            resolved_model,
            reasoning_effort,
        )
        payload = resolve_openai_payload_file_ids(payload, file_store)
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

    session_context = await _prepare_chat_responses_session_reuse(
        request,
        completion_request=completion_request,
        payload=payload,
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
        session_on_complete = _build_chat_session_completion_handler(
            request,
            session_context=session_context,
        )

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
                        if session_on_complete is not None:
                            await session_on_complete(response_obj)
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

    if completion_request.n == 1:
        await _store_chat_responses_session_state(
            request,
            session_context=session_context,
            upstream_response=upstream_results[0],
        )

    response_body = build_legacy_chat_completion_response(
        request=completion_request,
        upstream_results=upstream_results,
        response_model_name=resolved_model,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=response_body.model_dump())


async def _prepare_chat_responses_session_reuse(
    request: Request,
    *,
    completion_request: LegacyChatCompletionRequest,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    if completion_request.n != 1:
        return None

    session_store: ResponsesSessionStore | None = getattr(
        request.app.state,
        "responses_session_store",
        None,
    )
    raw_logger = getattr(request.app.state, "raw_io_logger", None)
    if session_store is None:
        return None

    session_key = _extract_chat_session_key(request, completion_request)
    if not session_key:
        return None

    payload["prompt_cache_key"] = session_key
    settings = getattr(request.app.state, "settings", None)
    prompt_cache_retention = getattr(settings, "default_prompt_cache_retention", None)
    if isinstance(prompt_cache_retention, str) and prompt_cache_retention.strip():
        payload["prompt_cache_retention"] = prompt_cache_retention.strip()

    input_items = payload.get("input")
    if not isinstance(input_items, list) or not input_items:
        return None

    canonical_full_input = _canonicalize_json_like(
        _normalize_chat_session_input_items(input_items)
    )
    context = {
        "session_key": session_key,
        "model": _string(payload.get("model")) or "",
        "instructions_hash": _hash_json_payload(payload.get("instructions")),
        "tools_hash": _hash_json_payload(payload.get("tools")),
        "tool_choice_hash": _hash_json_payload(payload.get("tool_choice")),
        "full_input": canonical_full_input,
        "reused": False,
    }

    previous_state = await session_store.get(session_key)
    if previous_state is None:
        return context

    if _should_prefer_chat_prompt_cache_strategy(session_key):
        if raw_logger is not None:
            raw_logger.log(
                "proxy.session_reuse",
                {
                    "path": request.url.path,
                    "session_key": session_key,
                    "reused": False,
                    "reason": "prompt_cache_preferred",
                    "input_count": len(canonical_full_input),
                    "previous_input_count": len(previous_state.full_input),
                },
            )
        return context

    if not _can_resume_chat_responses_session(previous_state, context):
        if raw_logger is not None:
            raw_logger.log(
                "proxy.session_reuse",
                {
                    "path": request.url.path,
                    "session_key": session_key,
                    "reused": False,
                    "reason": "state_mismatch",
                    "previous_model": previous_state.model,
                    "model": context["model"],
                },
            )
        return context

    previous_input = previous_state.full_input
    if len(canonical_full_input) <= len(previous_input):
        if raw_logger is not None:
            raw_logger.log(
                "proxy.session_reuse",
                {
                    "path": request.url.path,
                    "session_key": session_key,
                    "reused": False,
                    "reason": "no_appended_input",
                    "input_count": len(canonical_full_input),
                    "previous_input_count": len(previous_input),
                },
            )
        return context

    delta_input = _trim_replayed_assistant_input(
        canonical_full_input[len(previous_input):]
    )
    if not delta_input:
        if raw_logger is not None:
            raw_logger.log(
                "proxy.session_reuse",
                {
                    "path": request.url.path,
                    "session_key": session_key,
                    "reused": False,
                    "reason": "empty_delta_after_trim",
                    "input_count": len(canonical_full_input),
                    "previous_input_count": len(previous_input),
                },
            )
        return context

    payload["previous_response_id"] = previous_state.response_id
    payload["input"] = copy.deepcopy(delta_input)
    context["reused"] = True
    context["delta_input"] = delta_input
    context["previous_response_id"] = previous_state.response_id

    if raw_logger is not None:
        raw_logger.log(
            "proxy.session_reuse",
            {
                "path": request.url.path,
                "session_key": session_key,
                "reused": True,
                "previous_response_id": previous_state.response_id,
                "input_count": len(canonical_full_input),
                "previous_input_count": len(previous_input),
                "delta_input_count": len(delta_input),
            },
        )

    return context


def _build_chat_session_completion_handler(
    request: Request,
    *,
    session_context: dict[str, Any] | None,
):
    if not session_context:
        return None

    async def _on_complete(upstream_response: dict[str, Any]) -> None:
        await _store_chat_responses_session_state(
            request,
            session_context=session_context,
            upstream_response=upstream_response,
        )

    return _on_complete


async def _store_chat_responses_session_state(
    request: Request,
    *,
    session_context: dict[str, Any] | None,
    upstream_response: dict[str, Any],
) -> None:
    if not session_context:
        return

    response_id = _string(upstream_response.get("id"))
    if not response_id:
        return

    session_store: ResponsesSessionStore | None = getattr(
        request.app.state,
        "responses_session_store",
        None,
    )
    raw_logger = getattr(request.app.state, "raw_io_logger", None)
    if session_store is None:
        return

    await session_store.put(
        ResponsesSessionState(
            session_key=session_context["session_key"],
            response_id=response_id,
            model=session_context["model"],
            instructions_hash=session_context["instructions_hash"],
            tools_hash=session_context["tools_hash"],
            tool_choice_hash=session_context["tool_choice_hash"],
            full_input=copy.deepcopy(session_context["full_input"]),
            updated_at=time.time(),
        )
    )

    if raw_logger is not None:
        raw_logger.log(
            "proxy.session_state.updated",
            {
                "path": request.url.path,
                "session_key": session_context["session_key"],
                "response_id": response_id,
                "model": session_context["model"],
                "input_count": len(session_context["full_input"]),
                "reused": session_context.get("reused") is True,
            },
        )


def _can_resume_chat_responses_session(
    previous_state: ResponsesSessionState,
    context: dict[str, Any],
) -> bool:
    if previous_state.model != context["model"]:
        return False
    if previous_state.instructions_hash != context["instructions_hash"]:
        return False
    if previous_state.tools_hash != context["tools_hash"]:
        return False
    if previous_state.tool_choice_hash != context["tool_choice_hash"]:
        return False

    current_input = context["full_input"]
    previous_input = previous_state.full_input
    if len(current_input) < len(previous_input):
        return False
    return current_input[: len(previous_input)] == previous_input


def _trim_replayed_assistant_input(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = 0
    while index < len(input_items):
        item = input_items[index]
        if not _is_assistant_originated_item(item):
            break
        index += 1
    return input_items[index:]


def _is_assistant_originated_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    item_type = _string(item.get("type")).strip().lower()
    if item_type == "function_call":
        return True
    role = _string(item.get("role")).strip().lower()
    return role == "assistant"


def _extract_chat_session_key(
    request: Request,
    completion_request: LegacyChatCompletionRequest,
) -> str | None:
    header_value = request.headers.get(_NANOBOT_SESSION_HEADER)
    normalized = _normalize_chat_session_identity(header_value)
    if normalized:
        return normalized
    return _normalize_chat_user_identity(completion_request.user)


def _normalize_chat_session_identity(raw_value: Any) -> str | None:
    if not isinstance(raw_value, str):
        return None
    normalized = raw_value.strip()
    if not normalized:
        return None
    return f"nanobot:{normalized}"


def _normalize_chat_user_identity(raw_value: Any) -> str | None:
    if not isinstance(raw_value, str):
        return None
    normalized = raw_value.strip()
    if not normalized:
        return None
    prefix = "nanobot:"
    if not normalized.startswith(prefix):
        return None
    session_id = normalized[len(prefix):].strip()
    if not session_id:
        return None
    return f"nanobot:{session_id}"


def _should_prefer_chat_prompt_cache_strategy(session_key: str) -> bool:
    return session_key.startswith("nanobot:")


def _normalize_chat_session_input_items(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_items: list[dict[str, Any]] = []
    for item in input_items:
        if not isinstance(item, dict):
            normalized_items.append(item)
            continue

        normalized_item = dict(item)
        if _string(normalized_item.get("role")).strip().lower() == "user":
            content = normalized_item.get("content")
            if isinstance(content, list):
                normalized_item["content"] = _normalize_chat_session_content(content)
        normalized_items.append(normalized_item)
    return normalized_items


def _normalize_chat_session_content(content_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not content_items:
        return content_items

    normalized_content: list[dict[str, Any]] = []
    for index, part in enumerate(content_items):
        if index == 0:
            normalized_first = _normalize_nanobot_runtime_part(part)
            if normalized_first is None:
                continue
            normalized_content.append(normalized_first)
            continue
        normalized_content.append(part)
    return normalized_content


def _normalize_nanobot_runtime_part(part: Any) -> dict[str, Any] | None:
    if not isinstance(part, dict):
        return part

    if _string(part.get("type")).strip().lower() != "input_text":
        return part

    text = part.get("text")
    if not isinstance(text, str):
        return part

    normalized_text = _strip_nanobot_runtime_context(text)
    if normalized_text == text:
        return part
    if not normalized_text:
        return None

    normalized_part = dict(part)
    normalized_part["text"] = normalized_text
    return normalized_part


def _strip_nanobot_runtime_context(text: str) -> str:
    if not text.startswith(_NANOBOT_RUNTIME_CONTEXT_TAG):
        return text

    remainder = text[len(_NANOBOT_RUNTIME_CONTEXT_TAG):]
    separator = "\n\n"
    if separator not in remainder:
        return ""
    _, _, payload = remainder.partition(separator)
    return payload.lstrip()


def _json_compact(value: Any) -> str:
    return json.dumps(
        _canonicalize_json_like(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonicalize_json_like(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonicalize_json_like(value[key])
            for key in sorted(value.keys(), key=lambda item: str(item))
        }
    if isinstance(value, list):
        return [_canonicalize_json_like(item) for item in value]
    return value


def _hash_json_payload(value: Any) -> str | None:
    if value is None:
        return None
    try:
        encoded = _json_compact(value).encode("utf-8")
    except Exception:
        return None
    return hashlib.sha256(encoded).hexdigest()[:16]


def _string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ""
