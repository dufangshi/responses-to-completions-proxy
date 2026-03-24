from __future__ import annotations

import copy
import hashlib
import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.services.file_store import resolve_native_message_file_ids, resolve_openai_payload_file_ids
from app.services.responses_client import BaseResponsesGateway, UpstreamAPIError
from app.services.responses_session_store import ResponsesSessionState, ResponsesSessionStore
from app.services.session_reuse_fallback import (
    build_session_reuse_fallback_payload,
    build_stateless_tool_delta_input,
    log_session_reuse_fallback,
    mark_session_reuse_fallback_used,
    should_retry_session_reuse,
)
from app.services.streaming_adapter import iter_upstream_sse_events

router = APIRouter()


@router.post("/v1/messages")
@router.post("/messages")
@router.post("/v1/message")
@router.post("/message")
async def create_message(request: Request) -> Response:
    gateway: BaseResponsesGateway = request.app.state.responses_gateway
    settings = request.app.state.settings

    try:
        raw_payload = await request.json()
    except Exception:
        return _error_response(
            status.HTTP_400_BAD_REQUEST,
            "Request body must be valid JSON object.",
        )

    if not isinstance(raw_payload, dict):
        return _error_response(
            status.HTTP_400_BAD_REQUEST,
            "Request body must be a JSON object.",
        )

    if settings.upstream_mode == "messages":
        raw_payload = _prepare_native_message_request(raw_payload, request)
        if raw_payload.get("stream") is True:
            try:
                upstream_lines = await gateway.stream_native_message(raw_payload)
            except UpstreamAPIError as exc:
                return JSONResponse(
                    status_code=exc.status_code,
                    content=_anthropic_error_payload(exc.status_code, exc.payload),
                )

            return StreamingResponse(
                _raw_sse_passthrough(upstream_lines),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            upstream_response = await gateway.create_native_message(raw_payload)
        except UpstreamAPIError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content=_anthropic_error_payload(exc.status_code, exc.payload),
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=upstream_response,
        )

    try:
        payload, response_model_name = _build_responses_payload_from_messages_request(
            raw_payload,
            request,
        )
    except ValueError as exc:
        return _error_response(status.HTTP_400_BAD_REQUEST, str(exc))

    session_context = await _prepare_responses_session_reuse(
        request,
        raw_payload=raw_payload,
        payload=payload,
    )

    if payload.get("stream") is True:
        try:
            upstream_lines = await _stream_response_with_session_reuse_fallback(
                request,
                gateway=gateway,
                payload=payload,
                session_context=session_context,
            )
        except UpstreamAPIError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content=_anthropic_error_payload(exc.status_code, exc.payload),
            )

        return StreamingResponse(
            _anthropic_sse_passthrough(
                upstream_lines=upstream_lines,
                response_model_name=response_model_name,
                on_complete=_build_session_completion_handler(request, session_context),
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        upstream_response = await _create_response_with_session_reuse_fallback(
            request,
            gateway=gateway,
            payload=payload,
            session_context=session_context,
        )
    except UpstreamAPIError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=_anthropic_error_payload(exc.status_code, exc.payload),
        )

    await _store_responses_session_state(
        request,
        session_context=session_context,
        upstream_response=upstream_response,
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=_openai_response_to_anthropic_message(
            upstream_response,
            response_model_name=response_model_name,
        ),
    )


@router.post("/v1/messages/count_tokens")
@router.post("/messages/count_tokens")
async def count_message_tokens(request: Request) -> Response:
    gateway: BaseResponsesGateway = request.app.state.responses_gateway
    settings = request.app.state.settings

    try:
        raw_payload = await request.json()
    except Exception:
        return _error_response(
            status.HTTP_400_BAD_REQUEST,
            "Request body must be valid JSON object.",
        )

    if not isinstance(raw_payload, dict):
        return _error_response(
            status.HTTP_400_BAD_REQUEST,
            "Request body must be a JSON object.",
        )

    if settings.upstream_mode == "messages":
        native_payload = _prepare_native_message_request(raw_payload, request)
        native_payload.pop("stream", None)
        native_payload["max_tokens"] = 1

        try:
            upstream_response = await gateway.create_native_message(native_payload)
        except UpstreamAPIError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content=_anthropic_error_payload(exc.status_code, exc.payload),
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"input_tokens": _extract_anthropic_input_tokens(upstream_response)},
        )

    try:
        payload, _ = _build_responses_payload_from_messages_request(
            raw_payload,
            request,
        )
    except ValueError as exc:
        return _error_response(status.HTTP_400_BAD_REQUEST, str(exc))

    payload.pop("stream", None)
    payload["store"] = False
    payload["max_output_tokens"] = 1

    try:
        upstream_response = await gateway.create_response(payload)
    except UpstreamAPIError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=_anthropic_error_payload(exc.status_code, exc.payload),
        )

    usage = upstream_response.get("usage") if isinstance(upstream_response.get("usage"), dict) else {}
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"input_tokens": _as_int(usage.get("input_tokens"))},
    )


def _build_responses_payload_from_messages_request(
    raw_payload: dict[str, Any],
    request: Request,
) -> tuple[dict[str, Any], str]:
    settings = request.app.state.settings
    file_store = request.app.state.file_store

    resolved_model, reasoning_effort = settings.resolve_model_and_reasoning(None)

    input_items, system_text = _convert_messages_input(raw_payload)

    payload: dict[str, Any] = {
        "model": resolved_model,
        "input": input_items,
        "store": False,
    }

    prompt_cache_key = _extract_prompt_cache_key(raw_payload)
    if prompt_cache_key:
        payload["prompt_cache_key"] = prompt_cache_key
        if settings.default_prompt_cache_retention:
            payload["prompt_cache_retention"] = settings.default_prompt_cache_retention

    if system_text:
        payload["instructions"] = system_text
    elif "codex" in resolved_model.lower():
        payload["instructions"] = "You are a helpful assistant."

    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    max_tokens = _as_int(raw_payload.get("max_tokens"))
    if max_tokens > 0:
        payload["max_output_tokens"] = max_tokens

    if raw_payload.get("stream") is True:
        payload["stream"] = True

    temperature = raw_payload.get("temperature")
    if isinstance(temperature, (int, float)):
        payload["temperature"] = float(temperature)

    top_p = raw_payload.get("top_p")
    if isinstance(top_p, (int, float)):
        payload["top_p"] = float(top_p)

    stop_sequences = raw_payload.get("stop_sequences")
    if isinstance(stop_sequences, str) and stop_sequences.strip():
        payload["stop"] = stop_sequences
    elif isinstance(stop_sequences, list):
        stops = [item for item in stop_sequences if isinstance(item, str) and item]
        if stops:
            payload["stop"] = stops

    converted_tools = _convert_tools(raw_payload.get("tools"))
    if converted_tools:
        payload["tools"] = converted_tools

    converted_tool_choice = _convert_tool_choice(raw_payload.get("tool_choice"))
    if converted_tool_choice is not None:
        payload["tool_choice"] = converted_tool_choice

    resolved_payload = resolve_openai_payload_file_ids(payload, file_store)
    _log_cache_diagnostics(request, raw_payload=raw_payload, payload=resolved_payload)
    return resolved_payload, resolved_model


async def _create_response_with_session_reuse_fallback(
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


async def _stream_response_with_session_reuse_fallback(
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


async def _prepare_responses_session_reuse(
    request: Request,
    *,
    raw_payload: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    session_store: ResponsesSessionStore | None = getattr(
        request.app.state,
        "responses_session_store",
        None,
    )
    raw_logger = getattr(request.app.state, "raw_io_logger", None)
    if session_store is None:
        return None

    session_key = _extract_prompt_cache_key(raw_payload)
    if not session_key:
        return None

    input_items = payload.get("input")
    if not isinstance(input_items, list) or not input_items:
        return None

    canonical_full_input = _canonicalize_json_like(input_items)
    context = {
        "session_key": session_key,
        "model": _string(payload.get("model")) or "",
        "instructions_hash": _hash_json_payload(payload.get("instructions")),
        "tools_hash": _hash_json_payload(payload.get("tools")),
        "tool_choice_hash": _hash_json_payload(payload.get("tool_choice")),
        "full_input": canonical_full_input,
        "reused": False,
        "store_session_state": True,
    }

    previous_state = await session_store.get(session_key)
    if previous_state is None:
        return context

    if not _can_resume_responses_session(previous_state, context):
        context["store_session_state"] = False
        if raw_logger is not None:
            raw_logger.log(
                "proxy.session_reuse",
                {
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
                    "session_key": session_key,
                    "reused": False,
                    "reason": "no_appended_input",
                    "input_count": len(canonical_full_input),
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
                    "session_key": session_key,
                    "reused": False,
                    "reason": "tool_delta_stateless",
                    "input_count": len(canonical_full_input),
                    "previous_input_count": len(previous_input),
                    "delta_input_count": len(stateless_tool_delta),
                },
            )
        return context

    delta_input = _trim_replayed_assistant_input(appended_input)
    if not delta_input:
        if raw_logger is not None:
            raw_logger.log(
                "proxy.session_reuse",
                {
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
                "session_key": session_key,
                "reused": True,
                "previous_response_id": previous_state.response_id,
                "input_count": len(canonical_full_input),
                "previous_input_count": len(previous_input),
                "delta_input_count": len(delta_input),
            },
        )
    return context


def _can_resume_responses_session(
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


def _build_session_completion_handler(
    request: Request,
    session_context: dict[str, Any] | None,
):
    if not session_context:
        return None

    async def _on_complete(response_id: str) -> None:
        await _store_responses_session_state(
            request,
            session_context=session_context,
            upstream_response={"id": response_id},
        )

    return _on_complete


async def _store_responses_session_state(
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
            updated_at=time.time(),
        )
    )
    if raw_logger is not None:
        raw_logger.log(
            "proxy.session_state.updated",
            {
                "session_key": session_context["session_key"],
                "response_id": response_id,
                "model": session_context["model"],
                "input_count": len(session_context["full_input"]),
                "reused": session_context.get("reused") is True,
            },
        )


def _convert_messages_input(raw_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    messages = raw_payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be an array.")

    input_items: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be an object.")
        input_items.extend(_convert_message_to_input_items(message, index=index))

    system_text = _convert_system_text(raw_payload.get("system"))
    return input_items, system_text


def _convert_message_to_input_items(message: dict[str, Any], *, index: int) -> list[dict[str, Any]]:
    role = message.get("role")
    if not isinstance(role, str) or not role.strip():
        raise ValueError(f"messages[{index}].role must be a non-empty string.")

    role_value = role.strip().lower()
    if role_value not in {"user", "assistant"}:
        raise ValueError(f"messages[{index}].role must be user or assistant.")

    raw_content = message.get("content")
    blocks = _normalize_message_blocks(raw_content, message_index=index)

    results: list[dict[str, Any]] = []
    buffered_content: list[dict[str, Any]] = []

    def flush_buffer() -> None:
        if not buffered_content:
            return
        results.append({"role": role_value, "content": list(buffered_content)})
        buffered_content.clear()

    for block_index, block in enumerate(blocks):
        block_type = block.get("type")

        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"messages[{index}].content[{block_index}].text must be a string."
                )
            buffered_content.append(
                {
                    "type": "input_text" if role_value == "user" else "output_text",
                    "text": text,
                }
            )
            continue

        if block_type == "image":
            if role_value != "user":
                raise ValueError(
                    f"messages[{index}].content[{block_index}] image blocks are only supported for user messages."
                )
            image_url = _extract_image_url(block, message_index=index, block_index=block_index)
            buffered_content.append({"type": "input_image", "image_url": image_url})
            continue

        if block_type == "document":
            if role_value != "user":
                raise ValueError(
                    f"messages[{index}].content[{block_index}] document blocks are only supported for user messages."
                )
            buffered_content.append(
                _extract_input_file_from_document_block(
                    block,
                    message_index=index,
                    block_index=block_index,
                )
            )
            continue

        if block_type == "input_file":
            if role_value != "user":
                raise ValueError(
                    f"messages[{index}].content[{block_index}] input_file blocks are only supported for user messages."
                )
            buffered_content.append(
                _extract_input_file_block(
                    block,
                    message_index=index,
                    block_index=block_index,
                )
            )
            continue

        if block_type == "tool_use":
            if role_value != "assistant":
                raise ValueError(
                    f"messages[{index}].content[{block_index}] tool_use blocks require assistant role."
                )
            flush_buffer()
            tool_name = block.get("name")
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError(
                    f"messages[{index}].content[{block_index}].name must be a non-empty string."
                )
            call_id = block.get("id")
            if not isinstance(call_id, str) or not call_id.strip():
                call_id = f"call_{uuid.uuid4().hex}"
            arguments = block.get("input")
            results.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": tool_name.strip(),
                    "arguments": _json_compact(arguments if arguments is not None else {}),
                }
            )
            continue

        if block_type == "tool_result":
            if role_value != "user":
                raise ValueError(
                    f"messages[{index}].content[{block_index}] tool_result blocks require user role."
                )
            flush_buffer()
            tool_use_id = block.get("tool_use_id")
            if not isinstance(tool_use_id, str) or not tool_use_id.strip():
                raise ValueError(
                    f"messages[{index}].content[{block_index}].tool_use_id must be a non-empty string."
                )
            results.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_use_id,
                    "output": _convert_tool_result_output(block.get("content")),
                }
            )
            continue

        raise ValueError(
            f"messages[{index}].content[{block_index}].type '{block_type}' is not supported."
        )

    flush_buffer()
    if not results:
        results.append(
            {
                "role": role_value,
                "content": [
                    {
                        "type": "input_text" if role_value == "user" else "output_text",
                        "text": "",
                    }
                ],
            }
        )
    return results


def _normalize_message_blocks(raw_content: Any, *, message_index: int) -> list[dict[str, Any]]:
    if isinstance(raw_content, str):
        return [{"type": "text", "text": raw_content}]

    if not isinstance(raw_content, list):
        raise ValueError(
            f"messages[{message_index}].content must be a string or an array of content blocks."
        )

    blocks: list[dict[str, Any]] = []
    for item in raw_content:
        if isinstance(item, str):
            blocks.append({"type": "text", "text": item})
            continue
        if isinstance(item, dict):
            blocks.append(item)
            continue
        raise ValueError(f"messages[{message_index}].content entries must be objects.")
    return blocks


def _convert_system_text(raw_system: Any) -> str | None:
    if raw_system is None:
        return None
    if isinstance(raw_system, str):
        stripped = raw_system.strip()
        return stripped or None
    if not isinstance(raw_system, list):
        raise ValueError("system must be a string or an array.")

    parts: list[str] = []
    for index, block in enumerate(raw_system):
        if isinstance(block, str):
            if block.strip():
                parts.append(block.strip())
            continue
        if not isinstance(block, dict):
            raise ValueError(f"system[{index}] must be a string or an object.")
        block_type = block.get("type")
        if block_type != "text":
            raise ValueError(f"system[{index}].type '{block_type}' is not supported.")
        text = block.get("text")
        if not isinstance(text, str):
            raise ValueError(f"system[{index}].text must be a string.")
        if _should_skip_system_text_for_cache(text):
            continue
        if text.strip():
            parts.append(text.strip())

    joined = "\n\n".join(parts).strip()
    return joined or None


def _should_skip_system_text_for_cache(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return stripped.lower().startswith("x-anthropic-billing-header:")


def _extract_prompt_cache_key(raw_payload: dict[str, Any]) -> str | None:
    metadata = raw_payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    for field_name in ("user_id", "session_id"):
        raw_value = metadata.get(field_name)
        if not isinstance(raw_value, str):
            continue
        normalized = _normalize_prompt_cache_identity(raw_value)
        if normalized:
            return normalized

    for system_text in _iter_raw_system_texts(raw_payload.get("system")):
        normalized = _normalize_prompt_cache_identity(system_text)
        if normalized:
            return normalized

    billing_header = _extract_billing_header_value(raw_payload.get("system"))
    if billing_header:
        normalized = _normalize_prompt_cache_identity(billing_header)
        if normalized:
            return normalized
    return None


def _iter_raw_system_texts(raw_system: Any) -> list[str]:
    if isinstance(raw_system, str):
        stripped = raw_system.strip()
        return [stripped] if stripped else []
    if not isinstance(raw_system, list):
        return []

    parts: list[str] = []
    for block in raw_system:
        if isinstance(block, str):
            stripped = block.strip()
            if stripped:
                parts.append(stripped)
            continue
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return parts


def _extract_billing_header_value(raw_system: Any) -> str | None:
    for text in _iter_raw_system_texts(raw_system):
        stripped = text.strip()
        if not stripped.lower().startswith("x-anthropic-billing-header:"):
            continue
        _, _, suffix = stripped.partition(":")
        suffix = suffix.strip()
        if suffix:
            return suffix
    return None


def _normalize_prompt_cache_identity(raw_value: str) -> str | None:
    stripped = raw_value.strip()
    if not stripped:
        return None

    identity = _extract_session_id(stripped) or stripped
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    return f"claude-code:{digest}"


def _extract_session_id(raw_value: str) -> str | None:
    try:
        parsed = json.loads(raw_value)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        session_id = parsed.get("session_id")
        if isinstance(session_id, str) and session_id.strip():
            return session_id.strip()

    match = re.search(r'"session_id"\s*:\s*"([^"]+)"', raw_value)
    if not match:
        match = re.search(r"\bsession_id\b\s*[:=]\s*['\"]?([^,'\"}\s]+)", raw_value)
    if not match:
        return None
    candidate = match.group(1).strip()
    return candidate or None


def _extract_image_url(block: dict[str, Any], *, message_index: int, block_index: int) -> str:
    source = block.get("source")
    if not isinstance(source, dict):
        raise ValueError(
            f"messages[{message_index}].content[{block_index}].source must be an object."
        )

    source_type = source.get("type")
    if source_type == "base64":
        media_type = source.get("media_type")
        data = source.get("data")
        if not isinstance(media_type, str) or not media_type.strip():
            raise ValueError(
                f"messages[{message_index}].content[{block_index}].source.media_type must be a string."
            )
        if not isinstance(data, str) or not data.strip():
            raise ValueError(
                f"messages[{message_index}].content[{block_index}].source.data must be a base64 string."
            )
        return f"data:{media_type};base64,{data}"

    if source_type == "url":
        url = source.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError(
                f"messages[{message_index}].content[{block_index}].source.url must be a string."
            )
        return url.strip()

    raise ValueError(
        f"messages[{message_index}].content[{block_index}].source.type '{source_type}' is not supported."
    )


def _extract_input_file_from_document_block(
    block: dict[str, Any],
    *,
    message_index: int,
    block_index: int,
) -> dict[str, Any]:
    source = block.get("source")
    if not isinstance(source, dict):
        raise ValueError(
            f"messages[{message_index}].content[{block_index}].source must be an object."
        )

    source_type = source.get("type")
    filename = _document_filename(block)
    if source_type == "base64":
        media_type = source.get("media_type")
        data = source.get("data")
        if not isinstance(media_type, str) or not media_type.strip():
            raise ValueError(
                f"messages[{message_index}].content[{block_index}].source.media_type must be a string."
            )
        if media_type.strip().lower() != "application/pdf":
            raise ValueError(
                f"messages[{message_index}].content[{block_index}] only application/pdf documents are supported."
            )
        if not isinstance(data, str) or not data.strip():
            raise ValueError(
                f"messages[{message_index}].content[{block_index}].source.data must be a base64 string."
            )
        return {
            "type": "input_file",
            "filename": filename,
            "file_data": f"data:{media_type.strip()};base64,{data.strip()}",
        }

    if source_type == "url":
        url = source.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError(
                f"messages[{message_index}].content[{block_index}].source.url must be a string."
            )
        return {
            "type": "input_file",
            "filename": filename,
            "file_url": url.strip(),
        }

    if source_type == "file":
        file_id = source.get("file_id")
        if not isinstance(file_id, str) or not file_id.strip():
            raise ValueError(
                f"messages[{message_index}].content[{block_index}].source.file_id must be a string."
            )
        return {
            "type": "input_file",
            "filename": filename,
            "file_id": file_id.strip(),
        }

    raise ValueError(
        f"messages[{message_index}].content[{block_index}].source.type '{source_type}' is not supported."
    )


def _extract_input_file_block(
    block: dict[str, Any],
    *,
    message_index: int,
    block_index: int,
) -> dict[str, Any]:
    file_id = block.get("file_id")
    file_data = block.get("file_data")
    file_url = block.get("file_url")
    filename = block.get("filename")

    if isinstance(filename, str) and filename.strip():
        resolved_filename = filename.strip()
    else:
        resolved_filename = "document.pdf"

    if isinstance(file_id, str) and file_id.strip():
        return {
            "type": "input_file",
            "filename": resolved_filename,
            "file_id": file_id.strip(),
        }

    if isinstance(file_url, str) and file_url.strip():
        return {
            "type": "input_file",
            "filename": resolved_filename,
            "file_url": file_url.strip(),
        }

    if isinstance(file_data, str) and file_data.strip():
        return {
            "type": "input_file",
            "filename": resolved_filename,
            "file_data": file_data.strip(),
        }

    raise ValueError(
        f"messages[{message_index}].content[{block_index}] input_file blocks require file_id, file_url, or file_data."
    )


def _document_filename(block: dict[str, Any]) -> str:
    for key in ("title", "filename", "name"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "document.pdf"


def _convert_tool_result_output(raw_content: Any) -> str:
    if isinstance(raw_content, str):
        return raw_content
    if raw_content is None:
        return ""
    if isinstance(raw_content, list):
        parts: list[str] = []
        for item in raw_content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)
    return _json_compact(raw_content)


def _convert_tools(raw_tools: Any) -> list[dict[str, Any]]:
    if raw_tools is None:
        return []
    if not isinstance(raw_tools, list):
        raise ValueError("tools must be an array.")

    converted: list[dict[str, Any]] = []
    for index, tool in enumerate(raw_tools):
        if not isinstance(tool, dict):
            raise ValueError(f"tools[{index}] must be an object.")
        name = tool.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"tools[{index}].name must be a non-empty string.")
        parameters = tool.get("input_schema")
        if parameters is None:
            parameters = {"type": "object", "properties": {}}
        if not isinstance(parameters, dict):
            raise ValueError(f"tools[{index}].input_schema must be an object.")
        converted_tool: dict[str, Any] = {
            "type": "function",
            "name": name.strip(),
            "parameters": _canonicalize_json_like(parameters),
        }
        description = tool.get("description")
        if isinstance(description, str) and description.strip():
            converted_tool["description"] = description.strip()
        converted.append(converted_tool)
    converted.sort(key=lambda item: item["name"])
    return converted


def _convert_tool_choice(raw_tool_choice: Any) -> str | dict[str, Any] | None:
    if raw_tool_choice is None:
        return None

    if isinstance(raw_tool_choice, str):
        value = raw_tool_choice.strip().lower()
        if value in {"auto", "none"}:
            return value
        if value == "any":
            return {"type": "any"}
        if value == "tool":
            return {"type": "any"}
        raise ValueError("tool_choice must be auto, any, tool, or none.")

    if not isinstance(raw_tool_choice, dict):
        raise ValueError("tool_choice must be a string or an object.")

    choice_type = raw_tool_choice.get("type")
    if not isinstance(choice_type, str) or not choice_type.strip():
        raise ValueError("tool_choice.type must be a non-empty string.")

    normalized_type = choice_type.strip().lower()
    if normalized_type in {"auto", "none", "any"}:
        return {"type": normalized_type}
    if normalized_type == "tool":
        name = raw_tool_choice.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tool_choice.name is required when tool_choice.type='tool'.")
        return {"type": "tool", "name": name.strip()}
    raise ValueError("tool_choice.type must be auto, any, tool, or none.")


def _openai_response_to_anthropic_message(
    upstream_response: dict[str, Any],
    *,
    response_model_name: str,
) -> dict[str, Any]:
    response_id = _string(upstream_response.get("id")) or f"msg_{uuid.uuid4().hex}"
    usage = upstream_response.get("usage") if isinstance(upstream_response.get("usage"), dict) else {}
    content: list[dict[str, Any]] = []
    output = upstream_response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                content.extend(_extract_message_content_blocks(item.get("content")))
                continue
            if item_type == "function_call":
                call_id = _string(item.get("call_id")) or _string(item.get("id")) or f"call_{uuid.uuid4().hex}"
                content.append(
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": _string(item.get("name")) or "tool",
                        "input": _parse_json_object(item.get("arguments")),
                    }
                )
    if not content:
        output_text = upstream_response.get("output_text")
        if isinstance(output_text, str) and output_text:
            content.append({"type": "text", "text": output_text})

    return {
        "id": response_id,
        "type": "message",
        "role": "assistant",
        "model": response_model_name,
        "content": content,
        "stop_reason": _map_response_stop_reason(upstream_response),
        "stop_sequence": None,
        "usage": _anthropic_usage_from_openai_usage(usage),
    }


def _extract_message_content_blocks(raw_content: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_content, list):
        return []
    blocks: list[dict[str, Any]] = []
    for item in raw_content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "output_text":
            text = item.get("text")
            if isinstance(text, str):
                blocks.append({"type": "text", "text": text})
    return blocks


def _map_response_stop_reason(upstream_response: dict[str, Any]) -> str:
    output = upstream_response.get("output")
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict) and item.get("type") == "function_call":
                return "tool_use"

    incomplete = upstream_response.get("incomplete_details")
    if isinstance(incomplete, dict):
        reason = incomplete.get("reason")
        if reason == "max_output_tokens":
            return "max_tokens"
    return "end_turn"


def _anthropic_sse_passthrough(
    *,
    upstream_lines: AsyncIterator[str],
    response_model_name: str,
    on_complete=None,
) -> AsyncIterator[bytes]:
    async def generator() -> AsyncIterator[bytes]:
        response_id = f"msg_{uuid.uuid4().hex}"
        created_at = int(time.time())
        message_started = False
        current_usage = _anthropic_usage_from_openai_usage({})
        blocks_by_index: dict[int, dict[str, Any]] = {}
        blocks_by_item_id: dict[str, dict[str, Any]] = {}

        async def emit(event_name: str, payload: dict[str, Any]) -> AsyncIterator[bytes]:
            yield f"event: {event_name}\n".encode("utf-8")
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

        async def ensure_message_start(model_name: str | None = None) -> None:
            nonlocal message_started
            if message_started:
                return
            message_started = True
            async for chunk in emit(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": response_id,
                        "type": "message",
                        "role": "assistant",
                        "model": model_name or response_model_name,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": dict(current_usage),
                    },
                },
            ):
                yield chunk

        async for event in iter_upstream_sse_events(upstream_lines):
            if event == "[DONE]":
                break
            if not isinstance(event, dict):
                continue

            event_type = event.get("type")
            if not isinstance(event_type, str):
                continue

            if event_type == "response.failed":
                response_obj = event.get("response")
                error_obj = response_obj.get("error") if isinstance(response_obj, dict) else None
                async for chunk in emit(
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": _string(error_obj.get("message")) if isinstance(error_obj, dict) else "Upstream streaming failed.",
                        },
                    },
                ):
                    yield chunk
                return

            response_obj = event.get("response")
            if isinstance(response_obj, dict):
                upstream_id = response_obj.get("id")
                if isinstance(upstream_id, str) and upstream_id.strip():
                    response_id = upstream_id
                usage_obj = response_obj.get("usage")
                if isinstance(usage_obj, dict):
                    current_usage = _anthropic_usage_from_openai_usage(usage_obj)

            if event_type == "response.created":
                async for chunk in ensure_message_start(
                    _string(response_obj.get("model")) if isinstance(response_obj, dict) else None
                ):
                    yield chunk
                continue

            if not message_started:
                async for chunk in ensure_message_start():
                    yield chunk

            if event_type == "response.output_item.added":
                output_index = _as_int(event.get("output_index"))
                item = event.get("item")
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "message":
                    blocks_by_index[output_index] = {
                        "kind": "text",
                        "started": False,
                        "closed": False,
                    }
                    continue
                if item_type == "function_call":
                    call_id = _string(item.get("call_id")) or _string(item.get("id")) or f"call_{uuid.uuid4().hex}"
                    item_id = _string(item.get("id")) or call_id
                    state = {
                        "kind": "tool_use",
                        "item_id": item_id,
                        "call_id": call_id,
                        "name": _string(item.get("name")) or "tool",
                        "arguments": _string(item.get("arguments")),
                        "started": True,
                        "closed": False,
                        "saw_argument_delta": False,
                        "output_index": output_index,
                    }
                    blocks_by_index[output_index] = state
                    blocks_by_item_id[item_id] = state
                    async for chunk in emit(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": output_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": call_id,
                                "name": state["name"],
                                "input": {},
                            },
                        },
                    ):
                        yield chunk
                    continue

            if event_type == "response.output_text.delta":
                output_index = _as_int(event.get("output_index"))
                state = blocks_by_index.setdefault(
                    output_index,
                    {"kind": "text", "started": False, "closed": False},
                )
                if not state.get("started"):
                    state["started"] = True
                    async for chunk in emit(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": output_index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    ):
                        yield chunk
                delta = event.get("delta")
                if isinstance(delta, str) and delta:
                    async for chunk in emit(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": output_index,
                            "delta": {"type": "text_delta", "text": delta},
                        },
                    ):
                        yield chunk
                continue

            if event_type == "response.function_call_arguments.delta":
                item_id = _string(event.get("item_id"))
                state = blocks_by_item_id.get(item_id)
                if not state:
                    continue
                delta = event.get("delta")
                if not isinstance(delta, str) or not delta:
                    continue
                state["saw_argument_delta"] = True
                current_arguments = state.get("arguments")
                if not isinstance(current_arguments, str):
                    current_arguments = ""
                state["arguments"] = f"{current_arguments}{delta}"
                async for chunk in emit(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": int(state["output_index"]),
                        "delta": {"type": "input_json_delta", "partial_json": delta},
                    },
                ):
                    yield chunk
                continue

            if event_type == "response.function_call_arguments.done":
                item_id = _string(event.get("item_id"))
                state = blocks_by_item_id.get(item_id)
                if not state or state.get("closed"):
                    continue
                arguments = event.get("arguments")
                if (
                    not state.get("saw_argument_delta")
                    and isinstance(arguments, str)
                    and arguments.strip()
                ):
                    async for chunk in emit(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": int(state["output_index"]),
                            "delta": {"type": "input_json_delta", "partial_json": arguments},
                        },
                    ):
                        yield chunk
                async for chunk in emit(
                    "content_block_stop",
                    {
                        "type": "content_block_stop",
                        "index": int(state["output_index"]),
                    },
                ):
                    yield chunk
                state["closed"] = True
                continue

            if event_type == "response.output_item.done":
                output_index = _as_int(event.get("output_index"))
                item = event.get("item")
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                state = blocks_by_index.get(output_index)
                if item_type == "message" and state and not state.get("closed"):
                    if not state.get("started"):
                        state["started"] = True
                        async for chunk in emit(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": output_index,
                                "content_block": {"type": "text", "text": ""},
                            },
                        ):
                            yield chunk
                    async for chunk in emit(
                        "content_block_stop",
                        {
                            "type": "content_block_stop",
                            "index": output_index,
                        },
                    ):
                        yield chunk
                    state["closed"] = True
                continue

            if event_type == "response.completed":
                completed_response = response_obj if isinstance(response_obj, dict) else {}
                usage_obj = completed_response.get("usage")
                if isinstance(usage_obj, dict):
                    current_usage = _anthropic_usage_from_openai_usage(usage_obj)
                async for chunk in emit(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": _map_response_stop_reason(completed_response),
                            "stop_sequence": None,
                        },
                        "usage": dict(current_usage),
                    },
                ):
                    yield chunk
                async for chunk in emit("message_stop", {"type": "message_stop"}):
                    yield chunk
                if on_complete is not None and isinstance(response_id, str) and response_id.strip():
                    await on_complete(response_id)
                return

        if message_started:
            async for chunk in emit(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": dict(current_usage),
                },
            ):
                yield chunk
            async for chunk in emit("message_stop", {"type": "message_stop"}):
                yield chunk
            if on_complete is not None and isinstance(response_id, str) and response_id.strip():
                await on_complete(response_id)

    return generator()


def _anthropic_error_payload(status_code: int, payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        error_obj = payload.get("error")
        if isinstance(error_obj, dict):
            message = _string(error_obj.get("message")) or "Proxy request failed."
        else:
            message = json.dumps(payload, ensure_ascii=False)
    else:
        message = _string(payload) or "Proxy request failed."

    return {
        "type": "error",
        "error": {
            "type": "api_error" if status_code >= 500 else "invalid_request_error",
            "message": message,
        },
    }


def _anthropic_usage_from_openai_usage(usage: dict[str, Any]) -> dict[str, int]:
    input_details = usage.get("input_tokens_details")
    cached_tokens = 0
    if isinstance(input_details, dict):
        cached_tokens = _as_int(input_details.get("cached_tokens"))
    return {
        "input_tokens": _as_int(usage.get("input_tokens")),
        "output_tokens": _as_int(usage.get("output_tokens")),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached_tokens,
    }


def _extract_anthropic_input_tokens(upstream_response: dict[str, Any]) -> int:
    usage = upstream_response.get("usage")
    if not isinstance(usage, dict):
        return 0
    return _as_int(usage.get("input_tokens"))


async def _raw_sse_passthrough(upstream_lines: AsyncIterator[str]) -> AsyncIterator[bytes]:
    async for line in upstream_lines:
        yield f"{line}\n".encode("utf-8")


def _prepare_native_message_request(raw_payload: dict[str, Any], request: Request) -> dict[str, Any]:
    payload = resolve_native_message_file_ids(dict(raw_payload), request.app.state.file_store)
    if request.url.query:
        payload["__proxy_query_string"] = request.url.query

    forward_headers: dict[str, str] = {}
    for header_name in ("anthropic-beta", "anthropic-version"):
        header_value = request.headers.get(header_name)
        if isinstance(header_value, str) and header_value.strip():
            forward_headers[header_name] = header_value.strip()
    if forward_headers:
        payload["__proxy_forward_headers"] = forward_headers

    if "cache_control" in payload:
        return payload

    beta_flag = request.query_params.get("beta")
    if beta_flag is None:
        return payload

    if beta_flag.strip().lower() not in {"1", "true", "yes", "on"}:
        return payload

    # Claude Code commonly sends explicit block breakpoints sparingly. Add
    # top-level automatic caching so the full stable prefix can advance as the
    # conversation grows.
    payload["cache_control"] = {"type": "ephemeral"}
    return payload


def _error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {
                "type": "api_error" if status_code >= 500 else "invalid_request_error",
                "message": message,
            },
        },
    )


def _parse_json_object(raw_value: Any) -> dict[str, Any]:
    if isinstance(raw_value, dict):
        return raw_value
    if not isinstance(raw_value, str):
        return {}
    stripped = raw_value.strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return {"raw": raw_value}
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}


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


def _log_cache_diagnostics(request: Request, *, raw_payload: dict[str, Any], payload: dict[str, Any]) -> None:
    raw_logger = getattr(request.app.state, "raw_io_logger", None)
    if raw_logger is None:
        return

    input_items = payload.get("input")
    stable_prefix_input = input_items[:-1] if isinstance(input_items, list) and len(input_items) > 1 else input_items
    diagnostics = {
        "path": request.url.path,
        "prompt_cache_key": payload.get("prompt_cache_key"),
        "prompt_cache_retention": payload.get("prompt_cache_retention"),
        "input_count": len(input_items) if isinstance(input_items, list) else 0,
        "tools_count": len(payload.get("tools")) if isinstance(payload.get("tools"), list) else 0,
        "instructions_hash": _hash_json_payload(payload.get("instructions")),
        "tools_hash": _hash_json_payload(payload.get("tools")),
        "tool_choice_hash": _hash_json_payload(payload.get("tool_choice")),
        "full_input_hash": _hash_json_payload(input_items),
        "stable_prefix_hash": _hash_json_payload(stable_prefix_input),
        "raw_system_hash": _hash_json_payload(raw_payload.get("system")),
        "metadata_hash": _hash_json_payload(raw_payload.get("metadata")),
    }
    raw_logger.log("proxy.cache_diagnostics", diagnostics)


def _hash_json_payload(value: Any) -> str | None:
    if value is None:
        return None
    try:
        encoded = _json_compact(value).encode("utf-8")
    except Exception:
        return None
    return hashlib.sha256(encoded).hexdigest()[:16]


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return 0
        try:
            return int(stripped)
        except ValueError:
            return 0
    return 0


def _string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ""
