from __future__ import annotations

import copy
from typing import Any

from fastapi import Request

from app.services.responses_client import UpstreamAPIError

_SESSION_REUSE_FALLBACK_STATUS_CODES = {500, 502, 503, 504}
_MISSING_TOOL_CALL_ERROR_SNIPPETS = (
    "No tool call found for function call output",
    "No function call found for function call output",
)


def build_stateless_tool_delta_input(
    *,
    previous_input: list[dict[str, Any]] | Any,
    appended_input: list[dict[str, Any]] | Any,
) -> list[dict[str, Any]] | None:
    if not isinstance(appended_input, list) or not appended_input:
        return None

    prefix_length = 0
    while prefix_length < len(appended_input):
        item = appended_input[prefix_length]
        if not _is_assistant_originated_item(item):
            break
        prefix_length += 1

    if prefix_length >= len(appended_input):
        return None

    first_user_item = appended_input[prefix_length]
    if _item_type(first_user_item) != "function_call_output":
        return None

    if not any(_item_type(item) == "function_call_output" for item in appended_input[prefix_length:]):
        return None

    context_prefix: list[dict[str, Any]] = []
    if isinstance(previous_input, list) and previous_input:
        replay_start = _find_stateless_tool_replay_start(previous_input)
        context_prefix = copy.deepcopy(previous_input[replay_start:])

    replay_input = context_prefix + copy.deepcopy(appended_input)
    if not _can_replay_tool_outputs(replay_input):
        return None
    return replay_input


def should_force_full_input_replay(delta_input: list[dict[str, Any]] | Any) -> bool:
    if not isinstance(delta_input, list) or not delta_input:
        return False
    if _item_type(delta_input[0]) != "function_call_output":
        return False
    return not _can_replay_tool_outputs(delta_input)


def build_session_reuse_fallback_payload(
    payload: dict[str, Any],
    *,
    session_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(session_context, dict):
        return None
    if session_context.get("reused") is not True:
        return None
    if not payload.get("previous_response_id"):
        return None

    full_input = session_context.get("request_input")
    if not isinstance(full_input, list) or not full_input:
        full_input = session_context.get("full_input")
    if not isinstance(full_input, list) or not full_input:
        return None

    fallback_payload = copy.deepcopy(payload)
    fallback_payload.pop("previous_response_id", None)
    fallback_payload["input"] = copy.deepcopy(full_input)
    return fallback_payload


def should_retry_session_reuse(
    exc: UpstreamAPIError,
    *,
    session_context: dict[str, Any] | None,
    payload: dict[str, Any],
) -> bool:
    if not _is_retryable_session_reuse_error(exc):
        return False
    return build_session_reuse_fallback_payload(
        payload,
        session_context=session_context,
    ) is not None


def mark_session_reuse_fallback_used(session_context: dict[str, Any] | None) -> None:
    if not isinstance(session_context, dict):
        return
    session_context["reused"] = False
    session_context["session_reuse_fallback_used"] = True


def log_session_reuse_fallback(
    request: Request,
    *,
    session_context: dict[str, Any] | None,
    exc: UpstreamAPIError,
) -> None:
    if not isinstance(session_context, dict):
        return

    raw_logger = getattr(request.app.state, "raw_io_logger", None)
    if raw_logger is None:
        return

    full_input = session_context.get("full_input")
    raw_logger.log(
        "proxy.session_reuse_fallback",
        {
            "path": request.url.path,
            "session_key": session_context.get("session_key"),
            "previous_response_id": session_context.get("previous_response_id"),
            "status_code": exc.status_code,
            "retry_input_count": len(full_input) if isinstance(full_input, list) else 0,
        },
    )


def _is_assistant_originated_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    item_type = _item_type(item)
    if item_type == "function_call":
        return True
    role = _string(item.get("role")).strip().lower()
    return role == "assistant"


def _item_type(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    return _string(item.get("type")).strip().lower()


def _find_stateless_tool_replay_start(previous_input: list[dict[str, Any]]) -> int:
    for index in range(len(previous_input) - 1, -1, -1):
        if not _is_tool_item(previous_input[index]):
            return index
    return 0


def _is_tool_item(item: Any) -> bool:
    item_type = _item_type(item)
    return item_type in {"function_call", "function_call_output"}


def _can_replay_tool_outputs(input_items: list[dict[str, Any]]) -> bool:
    seen_function_calls: set[str] = set()
    saw_function_call_output = False

    for item in input_items:
        item_type = _item_type(item)
        call_id = _string(item.get("call_id")).strip()
        if item_type == "function_call":
            if call_id:
                seen_function_calls.add(call_id)
            continue
        if item_type != "function_call_output":
            continue
        saw_function_call_output = True
        if not call_id or call_id not in seen_function_calls:
            return False

    return saw_function_call_output


def _is_retryable_session_reuse_error(exc: UpstreamAPIError) -> bool:
    if exc.status_code in _SESSION_REUSE_FALLBACK_STATUS_CODES:
        return True
    if exc.status_code != 400:
        return False
    message = _extract_upstream_error_message(exc.payload)
    if not message:
        return False
    return any(snippet in message for snippet in _MISSING_TOOL_CALL_ERROR_SNIPPETS)


def _extract_upstream_error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        error_obj = payload.get("error")
        if isinstance(error_obj, dict):
            message = error_obj.get("message")
            if isinstance(message, str):
                return message
        message = payload.get("message")
        if isinstance(message, str):
            return message
    if isinstance(payload, str):
        return payload
    return ""


def _string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ""
