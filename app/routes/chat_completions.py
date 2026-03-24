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
from app.services.session_reuse_fallback import (
    build_session_reuse_fallback_payload,
    build_stateless_tool_delta_input,
    log_session_reuse_fallback,
    mark_session_reuse_fallback_used,
    should_retry_session_reuse,
)
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
            upstream_lines = await _stream_chat_response_with_session_reuse_fallback(
                request,
                gateway=gateway,
                payload=payload,
                session_context=session_context,
            )
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
            upstream_results.append(
                await _create_chat_response_with_session_reuse_fallback(
                    request,
                    gateway=gateway,
                    payload=payload,
                    session_context=session_context,
                )
            )
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


async def _create_chat_response_with_session_reuse_fallback(
    request: Request,
    *,
    gateway: BaseResponsesGateway,
    payload: dict[str, Any],
    session_context: dict[str, Any] | None,
) -> dict[str, Any]:
    try:
        return await gateway.create_response(payload)
    except UpstreamAPIError as exc:
        if not should_retry_session_reuse(
            exc,
            session_context=session_context,
            payload=payload,
        ):
            raise

        fallback_payload = build_session_reuse_fallback_payload(
            payload,
            session_context=session_context,
        )
        if fallback_payload is None:
            raise

        log_session_reuse_fallback(
            request,
            session_context=session_context,
            exc=exc,
        )
        mark_session_reuse_fallback_used(session_context)
        return await gateway.create_response(fallback_payload)


async def _stream_chat_response_with_session_reuse_fallback(
    request: Request,
    *,
    gateway: BaseResponsesGateway,
    payload: dict[str, Any],
    session_context: dict[str, Any] | None,
):
    try:
        return await gateway.stream_response(payload)
    except UpstreamAPIError as exc:
        if not should_retry_session_reuse(
            exc,
            session_context=session_context,
            payload=payload,
        ):
            raise

        fallback_payload = build_session_reuse_fallback_payload(
            payload,
            session_context=session_context,
        )
        if fallback_payload is None:
            raise

        log_session_reuse_fallback(
            request,
            session_context=session_context,
            exc=exc,
        )
        mark_session_reuse_fallback_used(session_context)
        return await gateway.stream_response(fallback_payload)


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

    input_items = payload.get("input")
    if not isinstance(input_items, list) or not input_items:
        return None

    canonical_full_input = _canonicalize_json_like(
        _normalize_chat_session_input_items(input_items)
    )
    canonical_match_input = _canonicalize_json_like(
        _build_chat_session_match_input(canonical_full_input)
    )
    context = {
        "model": _string(payload.get("model")) or "",
        "instructions_hash": _hash_json_payload(payload.get("instructions")),
        "tools_hash": _hash_json_payload(payload.get("tools")),
        "tool_choice_hash": _hash_json_payload(payload.get("tool_choice")),
        "full_input": canonical_full_input,
        "match_input": canonical_match_input,
        "source_fingerprint": _build_chat_request_source_fingerprint(request),
        "root_fingerprint": _build_chat_session_root_fingerprint(
            canonical_match_input,
            payload=payload,
        ),
        "reused": False,
        "store_session_state": True,
    }

    settings = getattr(request.app.state, "settings", None)
    session_key, session_key_source = await _resolve_chat_session_identity(
        request,
        completion_request=completion_request,
        session_store=session_store,
        context=context,
    )
    if not session_key:
        return None

    context["session_key"] = session_key
    context["session_key_source"] = session_key_source

    payload["prompt_cache_key"] = session_key
    prompt_cache_retention = getattr(settings, "default_prompt_cache_retention", None)
    if isinstance(prompt_cache_retention, str) and prompt_cache_retention.strip():
        payload["prompt_cache_retention"] = prompt_cache_retention.strip()

    previous_state = await session_store.get(session_key)
    if previous_state is None:
        return context

    if _should_prefer_chat_prompt_cache_strategy(session_key, session_key_source):
        frozen_context = _try_freeze_chat_prompt_cache_prefix(previous_state, context)
        if frozen_context is None:
            if raw_logger is not None:
                raw_logger.log(
                    "proxy.session_reuse",
                    {
                        "path": request.url.path,
                        "session_key": session_key,
                        "session_key_source": session_key_source,
                        "reused": False,
                        "reason": "prompt_cache_prefix_mismatch",
                        "input_count": len(canonical_full_input),
                        "previous_input_count": len(previous_state.full_input),
                    },
                )
            return context

        payload["input"] = copy.deepcopy(frozen_context["full_input"])
        context.update(frozen_context)
        if raw_logger is not None:
            raw_logger.log(
                "proxy.session_reuse",
                {
                    "path": request.url.path,
                    "session_key": session_key,
                    "session_key_source": session_key_source,
                    "reused": False,
                    "reason": "prompt_cache_preferred",
                    "input_count": len(context["full_input"]),
                    "previous_input_count": len(previous_state.full_input),
                    "delta_input_count": len(context.get("delta_input") or []),
                },
            )
        return context

    if not _can_resume_chat_responses_session(previous_state, context):
        context["store_session_state"] = False
        if raw_logger is not None:
            raw_logger.log(
                "proxy.session_reuse",
                {
                    "path": request.url.path,
                    "session_key": session_key,
                    "session_key_source": session_key_source,
                    "reused": False,
                    "reason": "state_mismatch",
                    "previous_model": previous_state.model,
                    "model": context["model"],
                },
            )
        return context

    previous_input = previous_state.full_input
    previous_match_input = _state_match_input(previous_state)
    current_match_input = context["match_input"]
    if len(current_match_input) <= len(previous_match_input):
        if raw_logger is not None:
            raw_logger.log(
                "proxy.session_reuse",
                {
                    "path": request.url.path,
                    "session_key": session_key,
                    "session_key_source": session_key_source,
                    "reused": False,
                    "reason": "no_appended_input",
                    "input_count": len(current_match_input),
                    "previous_input_count": len(previous_input),
                },
            )
        return context

    appended_input = canonical_full_input[len(previous_input) :]
    stateless_tool_delta = build_stateless_tool_delta_input(
        previous_input=previous_input,
        appended_input=appended_input,
    )
    if stateless_tool_delta is not None:
        payload["input"] = copy.deepcopy(stateless_tool_delta)
        payload.pop("previous_response_id", None)
        context["delta_input"] = stateless_tool_delta
        context["session_reuse_mode"] = "tool_delta_stateless"
        if raw_logger is not None:
            raw_logger.log(
                "proxy.session_reuse",
                {
                    "path": request.url.path,
                    "session_key": session_key,
                    "session_key_source": session_key_source,
                    "reused": False,
                    "reason": "tool_delta_stateless",
                    "input_count": len(current_match_input),
                    "previous_input_count": len(previous_input),
                    "delta_input_count": len(stateless_tool_delta),
                },
            )
        return context

    delta_start_index = _skip_replayed_assistant_prefix(
        canonical_full_input,
        start_index=len(previous_match_input),
    )
    delta_input = canonical_full_input[delta_start_index:]
    if not delta_input:
        if raw_logger is not None:
            raw_logger.log(
                "proxy.session_reuse",
                {
                    "path": request.url.path,
                    "session_key": session_key,
                    "session_key_source": session_key_source,
                    "reused": False,
                    "reason": "empty_delta_after_trim",
                    "input_count": len(current_match_input),
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
                "session_key_source": session_key_source,
                "reused": True,
                "previous_response_id": previous_state.response_id,
                "input_count": len(current_match_input),
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

    if session_context.get("store_session_state") is False:
        raw_logger = getattr(request.app.state, "raw_io_logger", None)
        if raw_logger is not None:
            raw_logger.log(
                "proxy.session_state.skipped",
                {
                    "path": request.url.path,
                    "session_key": session_context.get("session_key"),
                    "reason": "state_mismatch_branch",
                },
            )
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
            match_input=copy.deepcopy(session_context.get("match_input")),
            source_fingerprint=session_context.get("source_fingerprint"),
            root_fingerprint=session_context.get("root_fingerprint"),
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
    if not _is_chat_session_state_compatible(previous_state, context):
        return False

    current_input = context["match_input"]
    previous_input = _state_match_input(previous_state)
    if len(current_input) < len(previous_input):
        return False
    return current_input[: len(previous_input)] == previous_input


def _skip_replayed_assistant_prefix(
    input_items: list[dict[str, Any]],
    *,
    start_index: int = 0,
) -> int:
    index = max(0, start_index)
    while index < len(input_items):
        item = input_items[index]
        if not _is_assistant_originated_item(item):
            break
        index += 1
    return index


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


async def _resolve_chat_session_identity(
    request: Request,
    *,
    completion_request: LegacyChatCompletionRequest,
    session_store: ResponsesSessionStore,
    context: dict[str, Any],
) -> tuple[str | None, str | None]:
    explicit_session_key = _extract_chat_session_key(request, completion_request)
    if explicit_session_key:
        return explicit_session_key, "explicit"

    settings = getattr(request.app.state, "settings", None)
    if not getattr(settings, "enable_generic_chat_session_inference", True):
        return None, None

    inferred_state = await _infer_generic_chat_session_state(session_store, context)
    if inferred_state is not None:
        return inferred_state.session_key, "inferred"

    return _generate_inferred_chat_session_key(), "generated"


def _should_prefer_chat_prompt_cache_strategy(
    session_key: str,
    session_key_source: str | None,
) -> bool:
    if session_key.startswith("nanobot:"):
        return True
    return session_key_source in {"inferred", "generated"}


async def _infer_generic_chat_session_state(
    session_store: ResponsesSessionStore,
    context: dict[str, Any],
) -> ResponsesSessionState | None:
    current_match_input = context["match_input"]
    source_fingerprint = context.get("source_fingerprint")
    root_fingerprint = context.get("root_fingerprint")
    best_state: ResponsesSessionState | None = None
    best_score = -1

    for state in await session_store.list_states():
        if not _is_chat_session_state_compatible(state, context):
            continue
        if source_fingerprint and state.source_fingerprint and state.source_fingerprint != source_fingerprint:
            continue
        if root_fingerprint and state.root_fingerprint and state.root_fingerprint != root_fingerprint:
            continue

        previous_match_input = _state_match_input(state)
        if len(current_match_input) < len(previous_match_input):
            continue
        if current_match_input[: len(previous_match_input)] != previous_match_input:
            continue

        current_score = len(previous_match_input)
        if current_score > best_score:
            best_state = state
            best_score = current_score
            continue
        if current_score == best_score and best_state is not None:
            if state.updated_at > best_state.updated_at:
                best_state = state

    return best_state


def _is_chat_session_state_compatible(
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
    return True


def _try_freeze_chat_prompt_cache_prefix(
    previous_state: ResponsesSessionState,
    context: dict[str, Any],
) -> dict[str, Any] | None:
    if not _is_chat_session_state_compatible(previous_state, context):
        return None

    previous_match_input = _state_match_input(previous_state)
    current_match_input = context["match_input"]
    current_full_input = context["full_input"]

    if len(current_match_input) < len(previous_match_input):
        return None
    if current_match_input[: len(previous_match_input)] != previous_match_input:
        return None

    delta_start_index = len(previous_match_input)
    delta_input = current_full_input[delta_start_index:]
    delta_match_input = current_match_input[delta_start_index:]
    if not delta_input and len(current_match_input) > len(previous_match_input):
        return None

    frozen_full_input = copy.deepcopy(previous_state.full_input) + copy.deepcopy(delta_input)
    frozen_match_input = copy.deepcopy(previous_match_input) + copy.deepcopy(delta_match_input)
    return {
        "full_input": frozen_full_input,
        "match_input": frozen_match_input,
        "delta_input": delta_input,
        "reused": False,
    }


def _generate_inferred_chat_session_key() -> str:
    return f"chat:{uuid.uuid4().hex}"


def _build_chat_request_source_fingerprint(request: Request) -> str | None:
    forwarded_for = _first_header_value(request.headers.get("x-forwarded-for"))
    real_ip = _first_header_value(request.headers.get("x-real-ip"))
    client_host = ""
    if request.client is not None and isinstance(request.client.host, str):
        client_host = request.client.host.strip()
    user_agent = _string(request.headers.get("user-agent")).strip()
    authorization = _string(request.headers.get("authorization")).strip()
    if authorization:
        authorization = hashlib.sha256(authorization.encode("utf-8")).hexdigest()[:16]

    fingerprint_payload = {
        "forwarded_for": forwarded_for,
        "real_ip": real_ip,
        "client_host": client_host,
        "user_agent": user_agent,
        "authorization_hash": authorization,
    }
    fingerprint_hash = _hash_json_payload(fingerprint_payload)
    if not fingerprint_hash:
        return None
    return f"src:{fingerprint_hash}"


def _build_chat_session_root_fingerprint(
    match_input: list[dict[str, Any]],
    *,
    payload: dict[str, Any],
) -> str | None:
    stable_prefix = _extract_chat_root_prefix(match_input)
    root_payload = {
        "instructions": payload.get("instructions"),
        "tools": payload.get("tools"),
        "tool_choice": payload.get("tool_choice"),
        "prefix": stable_prefix,
    }
    digest = _hash_json_payload(root_payload)
    if not digest:
        return None
    return f"root:{digest}"


def _extract_chat_root_prefix(match_input: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stable_prefix: list[dict[str, Any]] = []
    for item in match_input:
        if _is_assistant_originated_item(item):
            break
        stable_prefix.append(item)
        if len(stable_prefix) >= 4:
            break
    if stable_prefix:
        return stable_prefix
    return match_input[: min(len(match_input), 4)]


def _build_chat_session_match_input(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_items: list[dict[str, Any]] = []
    for item in input_items:
        normalized_items.append(_normalize_chat_match_item(item))
    return normalized_items


def _normalize_chat_match_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item

    normalized_item: dict[str, Any] = {}
    for key, value in item.items():
        if key == "call_id":
            continue
        if key == "content" and isinstance(value, list):
            normalized_item[key] = [_normalize_chat_match_item(part) for part in value]
            continue
        normalized_item[key] = _normalize_chat_match_item(value)
    return normalized_item


def _state_match_input(state: ResponsesSessionState) -> list[dict[str, Any]]:
    if isinstance(state.match_input, list) and state.match_input:
        return state.match_input
    return _canonicalize_json_like(_build_chat_session_match_input(state.full_input))


def _first_header_value(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    head, _, _ = value.partition(",")
    return head.strip()


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
