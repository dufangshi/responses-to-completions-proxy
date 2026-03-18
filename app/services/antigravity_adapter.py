from __future__ import annotations

import json
import time
import uuid
from typing import Any

from app.services.model_limits import resolve_model_max_output_tokens


class AntigravityAdapterError(ValueError):
    pass


def build_antigravity_request_from_responses(payload: dict[str, Any]) -> dict[str, Any]:
    model = _string(payload.get("model"))
    if not model:
        raise AntigravityAdapterError("model is required.")

    messages, system_texts = _build_messages_and_system(payload.get("input"))
    instructions = _string(payload.get("instructions"))
    if instructions:
        system_texts.append(instructions)

    request_payload: dict[str, Any] = {
        "model": model,
        "messages": messages or [{"role": "user", "content": " "}],
        "max_tokens": _resolve_max_tokens(payload, model),
    }

    if system_texts:
        request_payload["system"] = "\n\n".join(part for part in system_texts if part.strip())

    speed = _string(payload.get("speed")).strip().lower()
    if speed:
        request_payload["speed"] = speed

    reasoning = _convert_reasoning(payload.get("reasoning"))
    if reasoning is not None:
        _apply_reasoning(request_payload, model, reasoning)

    tools = _convert_tools(payload.get("tools"))
    tool_choice = _convert_tool_choice(payload.get("tool_choice"))
    if tools:
        forced_tool_name = _extract_forced_tool_name(tool_choice)
        if forced_tool_name:
            filtered_tools = [
                tool
                for tool in tools
                if _string(tool.get("name")).strip() == forced_tool_name
            ]
            if filtered_tools:
                tools = filtered_tools
                tool_choice = {"type": "any"}
        request_payload["tools"] = tools

    if tool_choice is not None:
        request_payload["tool_choice"] = tool_choice

    if payload.get("stream") is True:
        request_payload["stream"] = True

    temperature = payload.get("temperature")
    if isinstance(temperature, (int, float)):
        request_payload["temperature"] = float(temperature)

    top_p = payload.get("top_p")
    if isinstance(top_p, (int, float)):
        request_payload["top_p"] = float(top_p)

    stop = payload.get("stop")
    if isinstance(stop, str) and stop:
        request_payload["stop_sequences"] = [stop]
    elif isinstance(stop, list):
        stops = [item for item in stop if isinstance(item, str) and item]
        if stops:
            request_payload["stop_sequences"] = stops

    return request_payload


def antigravity_message_to_openai_response(
    message_payload: dict[str, Any],
    *,
    model: str,
    request_id: str | None = None,
) -> dict[str, Any]:
    created_at = int(time.time())
    response_id = _string(message_payload.get("id")) or request_id or f"resp_{uuid.uuid4().hex}"
    output_text, tool_calls = _extract_message_content(message_payload.get("content"))

    output_items: list[dict[str, Any]] = []
    if output_text:
        output_items.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": output_text,
                        "annotations": [],
                    }
                ],
            }
        )

    output_items.extend(tool_calls)

    usage_obj = message_payload.get("usage")
    input_tokens = _as_int(usage_obj.get("input_tokens")) if isinstance(usage_obj, dict) else 0
    output_tokens = _as_int(usage_obj.get("output_tokens")) if isinstance(usage_obj, dict) else 0
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }

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

    stop_reason = _string(message_payload.get("stop_reason"))
    if stop_reason == "max_tokens":
        response_obj["incomplete_details"] = {"reason": "max_output_tokens"}
    return response_obj


def antigravity_error_to_openai_error(status_code: int, payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        error_obj = payload.get("error")
        if isinstance(error_obj, dict):
            message = _string(error_obj.get("message")) or "Antigravity upstream error."
            code = _string(error_obj.get("type")) or f"HTTP_{status_code}"
            return {
                "error": {
                    "message": message,
                    "type": "server_error" if status_code >= 500 else "invalid_request_error",
                    "param": None,
                    "code": code,
                }
            }

    message = _string(payload)
    if not message:
        message = "Antigravity upstream error."
    return {
        "error": {
            "message": message,
            "type": "server_error" if status_code >= 500 else "invalid_request_error",
            "param": None,
            "code": f"HTTP_{status_code}",
        }
    }


def _build_messages_and_system(raw_input: Any) -> tuple[list[dict[str, Any]], list[str]]:
    system_texts: list[str] = []
    messages: list[dict[str, Any]] = []

    def append_block(role: str, block: dict[str, Any]) -> None:
        if role not in {"user", "assistant"}:
            return
        if messages and messages[-1].get("role") == role:
            content = messages[-1].get("content")
            if isinstance(content, list):
                content.append(block)
                return
        messages.append({"role": role, "content": [block]})

    if isinstance(raw_input, str):
        append_block("user", {"type": "text", "text": raw_input})
        return _finalize_messages(messages), system_texts

    if not isinstance(raw_input, list):
        return _finalize_messages(messages), system_texts

    for item in raw_input:
        if not isinstance(item, dict):
            continue

        item_type = _string(item.get("type"))
        if item_type == "function_call":
            name = _string(item.get("name"))
            if not name:
                continue
            call_id = _string(item.get("call_id")) or _string(item.get("id")) or f"call_{uuid.uuid4().hex}"
            arguments = _parse_arguments(item.get("arguments"))
            append_block(
                "assistant",
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": arguments,
                },
            )
            continue

        if item_type == "function_call_output":
            call_id = _string(item.get("call_id")) or _string(item.get("id")) or f"call_{uuid.uuid4().hex}"
            append_block(
                "user",
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": _stringify_tool_result(item.get("output")),
                },
            )
            continue

        role = _string(item.get("role")).lower()
        if role in {"system", "developer"}:
            text = _extract_text(item.get("content"))
            if text:
                system_texts.append(text)
            continue

        if role not in {"user", "assistant"}:
            continue

        for block in _content_to_blocks(item.get("content")):
            append_block(role, block)

    return _finalize_messages(messages), system_texts


def _finalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    for message in messages:
        role = _string(message.get("role"))
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, list):
            continue
        if not content:
            continue
        finalized.append({"role": role, "content": content})
    return finalized


def _content_to_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        if content:
            return [{"type": "text", "text": content}]
        return []
    if not isinstance(content, list):
        return []

    blocks: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = _string(part.get("type"))
        if part_type in {"text", "input_text", "output_text"}:
            text = _string(part.get("text"))
            if text:
                blocks.append({"type": "text", "text": text})
            continue
        if part_type == "refusal":
            refusal = _string(part.get("refusal")) or _string(part.get("text"))
            if refusal:
                blocks.append({"type": "text", "text": refusal})
            continue
        if part_type == "tool_use":
            name = _string(part.get("name"))
            if not name:
                continue
            tool_id = _string(part.get("id")) or f"call_{uuid.uuid4().hex}"
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": _parse_arguments(part.get("input")),
                }
            )
            continue
        if part_type == "tool_result":
            tool_use_id = _string(part.get("tool_use_id")) or _string(part.get("id")) or f"call_{uuid.uuid4().hex}"
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": _stringify_tool_result(part.get("content")),
                }
            )
    return blocks


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = _string(item.get("text")) or _string(item.get("refusal"))
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _convert_tools(raw_tools: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_tools, list):
        return []

    tools: list[dict[str, Any]] = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        tool_type = _string(tool.get("type")).lower()
        if tool_type and tool_type != "function":
            continue

        function_obj = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = _string(tool.get("name")) or _string(function_obj.get("name"))
        if not name:
            continue

        description = _string(tool.get("description")) or _string(function_obj.get("description"))
        parameters = (
            function_obj.get("parameters")
            if isinstance(function_obj.get("parameters"), dict)
            else tool.get("parameters")
        )
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}

        entry: dict[str, Any] = {
            "name": name,
            "input_schema": parameters,
        }
        if description:
            entry["description"] = description
        tools.append(entry)
    return tools


def _convert_tool_choice(raw_tool_choice: Any) -> dict[str, Any] | None:
    if raw_tool_choice is None:
        return None

    if isinstance(raw_tool_choice, str):
        value = raw_tool_choice.strip().lower()
        if value == "auto":
            return {"type": "auto"}
        if value == "required":
            return {"type": "any"}
        if value == "none":
            return None
        return None

    if not isinstance(raw_tool_choice, dict):
        return None

    choice_type = _string(raw_tool_choice.get("type")).lower()
    if choice_type == "function":
        name = _string(raw_tool_choice.get("name"))
        if name:
            return {"type": "tool", "name": name}
        function_obj = raw_tool_choice.get("function")
        if isinstance(function_obj, dict):
            function_name = _string(function_obj.get("name"))
            if function_name:
                return {"type": "tool", "name": function_name}
        return {"type": "any"}

    if choice_type in {"tool", "auto", "any"}:
        result = {"type": choice_type}
        name = _string(raw_tool_choice.get("name"))
        if choice_type == "tool" and name:
            result["name"] = name
        return result

    return None


def _extract_forced_tool_name(tool_choice: dict[str, Any] | None) -> str | None:
    if not isinstance(tool_choice, dict):
        return None
    if _string(tool_choice.get("type")).strip().lower() != "tool":
        return None
    name = _string(tool_choice.get("name")).strip()
    return name or None


def _convert_reasoning(raw_reasoning: Any) -> dict[str, Any] | None:
    if not isinstance(raw_reasoning, dict):
        return None

    effort = _string(raw_reasoning.get("effort")).strip().lower()
    if not effort:
        return None

    return {"effort": effort}


def _apply_reasoning(
    request_payload: dict[str, Any],
    model: str,
    reasoning: dict[str, Any],
) -> None:
    effort = _string(reasoning.get("effort")).strip().lower()
    if not effort:
        return

    if _is_claude_effort_model(model):
        request_payload["output_config"] = {"effort": effort}
        return

    request_payload["reasoning"] = {"effort": effort}


def _is_claude_effort_model(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized in {
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-opus-4-5",
    }


def _resolve_max_tokens(payload: dict[str, Any], model: str) -> int:
    raw_value = payload.get("max_output_tokens")
    if raw_value is None:
        raw_value = payload.get("max_tokens")

    parsed_value = _as_int(raw_value)
    if parsed_value <= 0:
        parsed_value = 4096

    model_limit = resolve_model_max_output_tokens(model)
    if model_limit is not None:
        return min(parsed_value, model_limit)
    return parsed_value


def _extract_message_content(content: Any) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(content, list):
        return "", []

    text_chunks: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = _string(part.get("type"))
        if part_type == "text":
            text = _string(part.get("text"))
            if text:
                text_chunks.append(text)
            continue
        if part_type == "tool_use":
            name = _string(part.get("name"))
            if not name:
                continue
            call_id = _string(part.get("id")) or f"call_{uuid.uuid4().hex}"
            arguments = part.get("input")
            arguments_text = _json_compact(arguments if arguments is not None else {})
            tool_calls.append(
                {
                    "type": "function_call",
                    "id": call_id,
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments_text,
                }
            )
    return "".join(text_chunks), tool_calls


def _parse_arguments(raw_arguments: Any) -> dict[str, Any]:
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        stripped = raw_arguments.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {"raw": raw_arguments}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    return {}


def _stringify_tool_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


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


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ""
