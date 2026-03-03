from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from typing import Any


class GeminiAdapterError(ValueError):
    pass


_GEMINI_CONTENT_FILTER_REASONS = {
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
}


def build_gemini_request_from_responses(payload: dict[str, Any]) -> dict[str, Any]:
    request_payload: dict[str, Any] = {}

    instructions = payload.get("instructions")
    has_system_instruction = isinstance(instructions, str) and instructions.strip()
    if has_system_instruction:
        request_payload["systemInstruction"] = {
            "parts": [{"text": instructions.strip()}],
        }

    contents = _build_gemini_contents(payload.get("input"), skip_system=bool(has_system_instruction))
    if not contents:
        contents = [{"role": "user", "parts": [{"text": ""}]}]
    request_payload["contents"] = contents

    tools = _build_gemini_tools(payload.get("tools"))
    if tools:
        request_payload["tools"] = tools

    tool_config = _build_gemini_tool_config(payload.get("tool_choice"))
    if tool_config:
        request_payload["toolConfig"] = tool_config

    generation_config = _build_gemini_generation_config(payload)
    if generation_config:
        request_payload["generationConfig"] = generation_config

    return request_payload


def gemini_response_to_openai_response(
    gemini_payload: dict[str, Any],
    *,
    model: str,
    request_id: str | None = None,
) -> dict[str, Any]:
    candidate = _first_candidate(gemini_payload)
    output_items, output_text = _build_openai_output(candidate)
    usage = _convert_usage(gemini_payload.get("usageMetadata"))
    created_at = _to_epoch(gemini_payload.get("createTime"))
    response_id = _string_or_none(gemini_payload.get("responseId")) or request_id or f"resp_{uuid.uuid4().hex}"

    response_obj: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": model,
        "output": output_items,
        "output_text": output_text,
        "usage": usage,
    }

    incomplete_reason = _map_finish_reason_to_incomplete_reason(candidate.get("finishReason"))
    if incomplete_reason:
        response_obj["incomplete_details"] = {"reason": incomplete_reason}

    return response_obj


def build_openai_response_from_stream_state(
    *,
    model: str,
    response_id: str,
    created_at: int,
    full_text: str,
    tool_calls: list[dict[str, Any]],
    usage_metadata: dict[str, Any] | None,
    finish_reason: str | None,
) -> dict[str, Any]:
    output_items: list[dict[str, Any]] = []
    if full_text:
        output_items.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": full_text,
                        "annotations": [],
                    }
                ],
            }
        )

    for tool_call in tool_calls:
        output_items.append(tool_call)

    response_obj: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": model,
        "output": output_items,
        "output_text": full_text,
        "usage": _convert_usage(usage_metadata),
    }
    incomplete_reason = _map_finish_reason_to_incomplete_reason(finish_reason)
    if incomplete_reason:
        response_obj["incomplete_details"] = {"reason": incomplete_reason}
    return response_obj


def extract_stream_delta_text(current_text: str, previous_text: str) -> str:
    if not current_text:
        return ""
    if current_text.startswith(previous_text):
        return current_text[len(previous_text) :]
    return current_text


def extract_tool_calls_from_candidate(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(candidate, dict):
        return []
    content = candidate.get("content")
    if not isinstance(content, dict):
        return []
    parts = content.get("parts")
    if not isinstance(parts, list):
        return []
    calls: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        function_call = part.get("functionCall")
        if not isinstance(function_call, dict):
            continue
        name = _string_or_none(function_call.get("name"))
        if not name:
            continue
        args_value = function_call.get("args")
        arguments = _normalize_function_arguments(args_value)
        call_id = _string_or_none(function_call.get("id")) or f"call_{uuid.uuid4().hex}"
        calls.append(
            {
                "type": "function_call",
                "id": call_id,
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
            }
        )
    return calls


def extract_text_from_candidate(candidate: dict[str, Any]) -> str:
    if not isinstance(candidate, dict):
        return ""
    content = candidate.get("content")
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""
    chunks: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str):
            chunks.append(text)
    return "".join(chunks)


def first_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    return _first_candidate(payload)


def gemini_error_to_openai_error(status_code: int, payload: Any) -> dict[str, Any]:
    error_obj = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error_obj, dict):
        message = json.dumps(payload, ensure_ascii=False) if payload is not None else "Gemini upstream error."
        return {
            "error": {
                "message": message,
                "type": "server_error" if status_code >= 500 else "invalid_request_error",
                "param": None,
                "code": "gemini_upstream_error",
            }
        }

    message = _string_or_none(error_obj.get("message")) or "Gemini upstream error."
    status_text = _string_or_none(error_obj.get("status"))
    code = status_text or str(error_obj.get("code") or status_code)
    param = _extract_error_param(error_obj.get("details"))
    error_type = "server_error" if status_code >= 500 else "invalid_request_error"
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


def _build_gemini_contents(raw_input: Any, *, skip_system: bool) -> list[dict[str, Any]]:
    if isinstance(raw_input, str):
        return [{"role": "user", "parts": [{"text": raw_input}]}]
    if not isinstance(raw_input, list):
        return []

    contents: list[dict[str, Any]] = []
    call_name_by_id: dict[str, str] = {}
    for item in raw_input:
        if not isinstance(item, dict):
            continue

        item_type = _string_or_none(item.get("type"))
        if item_type == "function_call":
            function_call = _to_gemini_function_call(item)
            call_id = _string_or_none(item.get("call_id")) or _string_or_none(item.get("id"))
            if call_id:
                call_name_by_id[call_id] = function_call["name"]
            contents.append({"role": "model", "parts": [{"functionCall": function_call}]})
            continue

        if item_type == "function_call_output":
            function_response = _to_gemini_function_response(item, call_name_by_id)
            contents.append({"role": "user", "parts": [{"functionResponse": function_response}]})
            continue

        role = _string_or_none(item.get("role"))
        if not role:
            continue

        role_lower = role.lower()
        if skip_system and role_lower in {"system", "developer"}:
            continue

        parts = _extract_text_parts(item.get("content"))
        if not parts:
            continue
        gemini_role = "model" if role_lower == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": parts})

    return contents


def _extract_text_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]

    if not isinstance(content, list):
        return []

    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = _string_or_none(part.get("type"))

        if part_type in {"input_text", "output_text", "text"}:
            text = _string_or_none(part.get("text"))
            if text is not None:
                parts.append({"text": text})
            continue

        if part_type == "refusal":
            refusal = _string_or_none(part.get("refusal")) or _string_or_none(part.get("text"))
            if refusal is not None:
                parts.append({"text": refusal})
            continue

        if part_type in {"input_image", "image_url"}:
            raise GeminiAdapterError("Gemini adapter currently does not support image content conversion.")

        text_fallback = _string_or_none(part.get("text"))
        if text_fallback is not None:
            parts.append({"text": text_fallback})

    return parts


def _to_gemini_function_call(item: dict[str, Any]) -> dict[str, Any]:
    name = _string_or_none(item.get("name"))
    if not name:
        raise GeminiAdapterError("function_call item requires a non-empty name.")
    arguments = _normalize_function_arguments(item.get("arguments"))
    try:
        args_value = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as exc:
        raise GeminiAdapterError(f"function_call.arguments must be valid JSON string: {exc.msg}") from exc
    if not isinstance(args_value, dict):
        args_value = {"value": args_value}
    return {"name": name, "args": args_value}


def _to_gemini_function_response(
    item: dict[str, Any],
    call_name_by_id: dict[str, str],
) -> dict[str, Any]:
    output = item.get("output")
    response_value: Any
    if isinstance(output, str):
        stripped = output.strip()
        if stripped:
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                response_value = {"output": output}
            else:
                response_value = parsed
        else:
            response_value = {"output": ""}
    elif output is None:
        response_value = {"output": ""}
    else:
        response_value = output

    if not isinstance(response_value, dict):
        response_value = {"output": response_value}

    name = _string_or_none(item.get("name"))
    if not name:
        call_id = _string_or_none(item.get("call_id"))
        if call_id:
            name = call_name_by_id.get(call_id)
    if not name:
        raise GeminiAdapterError("function_call_output item requires function name or a known call_id.")

    return {"name": name, "response": response_value}


def _build_gemini_tools(raw_tools: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_tools, list):
        return []
    declarations: list[dict[str, Any]] = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        tool_type = _string_or_none(tool.get("type"))
        if tool_type != "function":
            continue

        function_name = _string_or_none(tool.get("name"))
        function_description = _string_or_none(tool.get("description"))
        function_parameters = tool.get("parameters")

        function_obj = tool.get("function")
        if isinstance(function_obj, dict):
            function_name = function_name or _string_or_none(function_obj.get("name"))
            function_description = function_description or _string_or_none(function_obj.get("description"))
            if isinstance(function_obj.get("parameters"), dict):
                function_parameters = function_obj.get("parameters")

        if not function_name:
            raise GeminiAdapterError("tools[].function.name is required.")

        declaration: dict[str, Any] = {"name": function_name}
        if function_description:
            declaration["description"] = function_description
        if isinstance(function_parameters, dict):
            declaration["parameters"] = function_parameters
        declarations.append(declaration)

    if not declarations:
        return []
    return [{"functionDeclarations": declarations}]


def _build_gemini_tool_config(raw_tool_choice: Any) -> dict[str, Any] | None:
    if raw_tool_choice is None:
        return None

    mode = "AUTO"
    allowed_names: list[str] | None = None

    if isinstance(raw_tool_choice, str):
        lowered = raw_tool_choice.strip().lower()
        if lowered == "required":
            mode = "ANY"
        elif lowered == "none":
            mode = "NONE"
        elif lowered == "auto":
            mode = "AUTO"
    elif isinstance(raw_tool_choice, dict):
        tool_type = _string_or_none(raw_tool_choice.get("type"))
        if tool_type == "function":
            function_name = _string_or_none(raw_tool_choice.get("name"))
            function_obj = raw_tool_choice.get("function")
            if not function_name and isinstance(function_obj, dict):
                function_name = _string_or_none(function_obj.get("name"))
            if function_name:
                mode = "ANY"
                allowed_names = [function_name]

    config: dict[str, Any] = {"mode": mode}
    if allowed_names:
        config["allowedFunctionNames"] = allowed_names
    return {"functionCallingConfig": config}


def _build_gemini_generation_config(payload: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    max_output_tokens = payload.get("max_output_tokens")
    if isinstance(max_output_tokens, int):
        config["maxOutputTokens"] = max_output_tokens

    temperature = payload.get("temperature")
    if isinstance(temperature, (int, float)):
        config["temperature"] = float(temperature)

    top_p = payload.get("top_p")
    if isinstance(top_p, (int, float)):
        config["topP"] = float(top_p)

    stop_sequences = payload.get("stop")
    if isinstance(stop_sequences, str):
        config["stopSequences"] = [stop_sequences]
    elif isinstance(stop_sequences, list):
        normalized_stops = [value for value in stop_sequences if isinstance(value, str)]
        if normalized_stops:
            config["stopSequences"] = normalized_stops

    return config


def _build_openai_output(candidate: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    text = extract_text_from_candidate(candidate)
    tool_calls = extract_tool_calls_from_candidate(candidate)

    output: list[dict[str, Any]] = []
    if text:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        )

    output.extend(tool_calls)
    return output, text


def _convert_usage(raw_usage: Any) -> dict[str, int]:
    usage = raw_usage if isinstance(raw_usage, dict) else {}
    prompt_tokens = int(usage.get("promptTokenCount", 0) or 0)
    completion_tokens = int(usage.get("candidatesTokenCount", 0) or 0)
    total_tokens = int(usage.get("totalTokenCount", 0) or 0)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _map_finish_reason_to_incomplete_reason(finish_reason: Any) -> str | None:
    if not isinstance(finish_reason, str):
        return None
    normalized = finish_reason.strip().upper()
    if normalized == "MAX_TOKENS":
        return "max_output_tokens"
    if normalized in _GEMINI_CONTENT_FILTER_REASONS:
        return "content_filter"
    return None


def _first_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict):
                return candidate
    return {}


def _normalize_function_arguments(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "{}"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _extract_error_param(details: Any) -> str | None:
    if not isinstance(details, list):
        return None
    for detail in details:
        if not isinstance(detail, dict):
            continue
        violations = detail.get("fieldViolations")
        if not isinstance(violations, list):
            continue
        for violation in violations:
            if not isinstance(violation, dict):
                continue
            field = _string_or_none(violation.get("field"))
            if field:
                return field
    return None


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _to_epoch(raw_time: Any) -> int:
    if isinstance(raw_time, str):
        try:
            return int(datetime.fromisoformat(raw_time.replace("Z", "+00:00")).timestamp())
        except Exception:
            pass
    return int(time.time())
