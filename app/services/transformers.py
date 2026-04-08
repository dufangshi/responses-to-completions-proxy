from __future__ import annotations

import time
import uuid
from typing import Any

from app.models.legacy_chat_completions import (
    ChatCompletionChoice,
    ChatCompletionMessageOut,
    ChatCompletionUsage,
    LegacyChatCompletionRequest,
    LegacyChatCompletionResponse,
)
from app.models.legacy_completions import (
    CompletionChoice,
    CompletionUsage,
    LegacyCompletionRequest,
    LegacyCompletionResponse,
)
from app.services.responses_payload import ResponsesValidationError, validate_responses_input_items
from app.services.file_store import normalize_user_supplied_filename


class UnsupportedParameterError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.param = param
        self.code = code


UNSUPPORTED_FIELDS = (
    "suffix",
    "best_of",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "seed",
)

UNSUPPORTED_CHAT_FIELDS = (
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "seed",
    "response_format",
)


def build_responses_payload(
    request: LegacyCompletionRequest,
    target_model: str,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    _validate_supported_params(request)

    if isinstance(request.prompt, list):
        if len(request.prompt) > 1:
            raise UnsupportedParameterError(
                "prompt as list with multiple entries is not supported yet."
            )
        prompt_value = request.prompt[0]
    else:
        prompt_value = request.prompt

    payload: dict[str, Any] = {
        "model": target_model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt_value}],
            }
        ],
        "store": False,
    }
    payload.update(
        _build_shared_sampling_params(
            target_model=target_model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_logprobs=request.logprobs,
        )
    )
    if request.stream:
        payload["stream"] = True
    if _requires_non_empty_instructions(target_model):
        payload["instructions"] = "You are a helpful assistant."
    _attach_reasoning(payload, reasoning_effort)
    return payload


def build_legacy_completion_response(
    request: LegacyCompletionRequest,
    upstream_results: list[dict[str, Any]],
    response_model_name: str,
) -> LegacyCompletionResponse:
    choices: list[CompletionChoice] = []
    usage = CompletionUsage()
    for index, result in enumerate(upstream_results):
        output_text = extract_output_text(result)
        if request.echo:
            prompt_text = request.prompt[0] if isinstance(request.prompt, list) else request.prompt
            output_text = f"{prompt_text}{output_text}"

        output_text, finish_reason = apply_stop_sequences(
            text=output_text,
            stop=request.stop,
            default_finish_reason=map_finish_reason(result),
        )

        choices.append(
            CompletionChoice(
                text=output_text,
                index=index,
                logprobs=None,
                finish_reason=finish_reason,
            )
        )
        usage.prompt_tokens += int(result.get("usage", {}).get("input_tokens", 0))
        usage.completion_tokens += int(result.get("usage", {}).get("output_tokens", 0))
        usage.total_tokens += int(result.get("usage", {}).get("total_tokens", 0))

    response_id = upstream_results[0].get("id") if upstream_results else None
    created = int(upstream_results[0].get("created_at", time.time())) if upstream_results else int(time.time())
    return LegacyCompletionResponse(
        id=response_id or f"cmpl_{uuid.uuid4().hex}",
        created=created,
        model=response_model_name,
        choices=choices,
        usage=usage,
    )


def build_chat_responses_payload(
    request: LegacyChatCompletionRequest,
    target_model: str,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    _validate_supported_chat_params(request)
    input_items: list[dict[str, Any]] = []
    for message in request.messages:
        input_items.extend(build_chat_input_items(message.model_dump()))
    try:
        validate_responses_input_items(input_items)
    except ResponsesValidationError as exc:
        raise UnsupportedParameterError(
            str(exc),
            param=exc.param,
            code=exc.code,
        ) from exc
    payload: dict[str, Any] = {
        "model": target_model,
        "input": input_items,
        "store": False,
        **_build_shared_sampling_params(
            target_model=target_model,
            max_tokens=resolve_chat_max_tokens(request),
            temperature=request.temperature,
            top_p=request.top_p,
            top_logprobs=request.top_logprobs,
        ),
    }
    if request.tools is not None:
        payload["tools"] = _convert_chat_tools(request.tools)
    if request.tool_choice is not None:
        payload["tool_choice"] = _convert_chat_tool_choice(request.tool_choice)
    if request.stream:
        payload["stream"] = True
    instructions = _extract_chat_instructions(request)
    if instructions:
        payload["instructions"] = instructions
    elif _requires_non_empty_instructions(target_model):
        payload["instructions"] = "You are a helpful assistant."
    _attach_reasoning(payload, reasoning_effort)
    return payload


def build_legacy_chat_completion_response(
    request: LegacyChatCompletionRequest,
    upstream_results: list[dict[str, Any]],
    response_model_name: str,
) -> LegacyChatCompletionResponse:
    choices: list[ChatCompletionChoice] = []
    usage = ChatCompletionUsage()
    for index, result in enumerate(upstream_results):
        output_text = extract_output_text(result)
        tool_calls = extract_tool_calls(result)
        finish_reason = map_finish_reason(result)
        if finish_reason != "tool_calls":
            output_text, finish_reason = apply_stop_sequences(
                text=output_text,
                stop=request.stop,
                default_finish_reason=finish_reason,
            )
        message_content: str | None = output_text
        if tool_calls and not output_text:
            message_content = None
        choices.append(
            ChatCompletionChoice(
                index=index,
                message=ChatCompletionMessageOut(
                    content=message_content,
                    tool_calls=tool_calls or None,
                ),
                logprobs=None,
                finish_reason=finish_reason,
            )
        )
        usage.prompt_tokens += int(result.get("usage", {}).get("input_tokens", 0))
        usage.completion_tokens += int(result.get("usage", {}).get("output_tokens", 0))
        usage.total_tokens += int(result.get("usage", {}).get("total_tokens", 0))

    response_id = upstream_results[0].get("id") if upstream_results else None
    created = int(upstream_results[0].get("created_at", time.time())) if upstream_results else int(time.time())
    return LegacyChatCompletionResponse(
        id=response_id or f"chatcmpl_{uuid.uuid4().hex}",
        created=created,
        model=response_model_name,
        choices=choices,
        usage=usage,
    )


def build_chat_input_items(raw_message: dict[str, Any]) -> list[dict[str, Any]]:
    role = raw_message.get("role")
    if not isinstance(role, str):
        raise UnsupportedParameterError("messages[].role must be a string.")
    role_lower = role.lower()

    if role_lower == "tool":
        return [_build_tool_output_item(raw_message)]

    items: list[dict[str, Any]] = []

    if role_lower in {"assistant", "function"}:
        tool_call_items = _extract_assistant_tool_call_items(raw_message)
        if tool_call_items:
            items.extend(tool_call_items)

    normalized_role = role_lower if role_lower != "function" else "assistant"
    content_items = _build_message_content_items(raw_message, normalized_role)
    if content_items:
        built: dict[str, Any] = {"role": normalized_role, "content": content_items}
        if isinstance(raw_message.get("name"), str):
            built["name"] = raw_message["name"]
        items.insert(0, built)

    if not items:
        # Keep turns parseable for empty assistant/system messages.
        items.append(
            {
                "role": normalized_role,
                "content": [{"type": _text_type_for_role(normalized_role), "text": ""}],
            }
        )
    return items


def _build_message_content_items(raw_message: dict[str, Any], role: str) -> list[dict[str, Any]]:
    content = raw_message.get("content")
    if isinstance(content, str):
        return [{"type": _text_type_for_role(role), "text": content}]
    if isinstance(content, list):
        return [convert_chat_content_part(part, role) for part in content]
    if content is None:
        return []
    raise UnsupportedParameterError("messages[].content must be null, string, or content parts array.")


def convert_chat_content_part(part: Any, role: str) -> dict[str, Any]:
    if not isinstance(part, dict):
        raise UnsupportedParameterError("messages[].content[] entries must be objects.")

    part_type = part.get("type")
    if part_type is None and isinstance(part.get("text"), str):
        return {"type": _text_type_for_role(role), "text": part["text"]}

    if part_type == "text":
        text = part.get("text")
        if not isinstance(text, str):
            raise UnsupportedParameterError("messages[].content[].text must be a string.")
        return {"type": _text_type_for_role(role), "text": text}

    if part_type == "input_text":
        text = part.get("text")
        if not isinstance(text, str):
            raise UnsupportedParameterError("messages[].content[].text must be a string.")
        return {"type": _text_type_for_role(role), "text": text}

    if part_type == "output_text":
        text = part.get("text")
        if not isinstance(text, str):
            raise UnsupportedParameterError("messages[].content[].text must be a string.")
        return {"type": _text_type_for_role(role), "text": text}

    if part_type == "refusal":
        refusal = part.get("refusal")
        if refusal is None:
            refusal = part.get("text")
        if not isinstance(refusal, str):
            raise UnsupportedParameterError("messages[].content[].refusal must be a string.")
        if role != "assistant":
            raise UnsupportedParameterError("messages[].content[].type='refusal' is only valid for assistant role.")
        return {"type": "refusal", "refusal": refusal}

    if part_type == "image_url":
        if _text_type_for_role(role) != "input_text":
            raise UnsupportedParameterError("image content is only supported for user/system/developer messages.")
        image_field = part.get("image_url")
        image_url = image_field.get("url") if isinstance(image_field, dict) else image_field
        if not isinstance(image_url, str):
            raise UnsupportedParameterError("messages[].content[].image_url must contain a url string.")
        return {"type": "input_image", "image_url": image_url}

    if part_type == "input_image":
        if _text_type_for_role(role) != "input_text":
            raise UnsupportedParameterError("image content is only supported for user/system/developer messages.")
        image_url = part.get("image_url")
        if not isinstance(image_url, str):
            raise UnsupportedParameterError("messages[].content[].image_url must be a string.")
        built = {"type": "input_image", "image_url": image_url}
        if "detail" in part:
            built["detail"] = part["detail"]
        return built

    if part_type == "file":
        if _text_type_for_role(role) != "input_text":
            raise UnsupportedParameterError("file content is only supported for user/system/developer messages.")
        file_obj = part.get("file")
        if not isinstance(file_obj, dict):
            raise UnsupportedParameterError("messages[].content[].file must be an object.")
        return _build_input_file_part(file_obj)

    if part_type == "input_file":
        if _text_type_for_role(role) != "input_text":
            raise UnsupportedParameterError("file content is only supported for user/system/developer messages.")
        return _build_input_file_part(part)

    raise UnsupportedParameterError(
        f"messages[].content[] type '{part_type}' is not supported by this compatibility proxy."
    )


def _build_input_file_part(raw_file: dict[str, Any]) -> dict[str, Any]:
    resolved_filename = normalize_user_supplied_filename(
        raw_file.get("filename"),
        fallback="upload.bin",
    )

    file_id = raw_file.get("file_id")
    if isinstance(file_id, str) and file_id.strip():
        return {
            "type": "input_file",
            "filename": resolved_filename,
            "file_id": file_id.strip(),
        }

    file_url = raw_file.get("file_url")
    if isinstance(file_url, str) and file_url.strip():
        return {
            "type": "input_file",
            "filename": resolved_filename,
            "file_url": file_url.strip(),
        }

    file_data = raw_file.get("file_data")
    if isinstance(file_data, str) and file_data.strip():
        return {
            "type": "input_file",
            "filename": resolved_filename,
            "file_data": file_data.strip(),
        }

    raise UnsupportedParameterError(
        "messages[].content[].file requires file_id, file_url, or file_data."
    )


def extract_output_text(upstream_response: dict[str, Any]) -> str:
    chunks: list[str] = []
    output = upstream_response.get("output", [])
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for content_item in content:
                if isinstance(content_item, dict) and content_item.get("type") == "output_text":
                    text = content_item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
    if chunks:
        return "".join(chunks)
    output_text = upstream_response.get("output_text")
    if isinstance(output_text, str):
        return output_text
    return ""


def extract_tool_calls(upstream_response: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    output = upstream_response.get("output", [])
    if not isinstance(output, list):
        return calls

    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call":
            continue
        function_name = item.get("name")
        if not isinstance(function_name, str) or not function_name.strip():
            continue
        call_id = item.get("call_id")
        if not isinstance(call_id, str) or not call_id.strip():
            item_id = item.get("id")
            if isinstance(item_id, str) and item_id.strip():
                call_id = item_id
            else:
                call_id = f"call_{uuid.uuid4().hex}"
        arguments = item.get("arguments")
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": arguments if isinstance(arguments, str) else "",
                },
            }
        )
    return calls


def map_finish_reason(upstream_response: dict[str, Any]) -> str:
    if extract_tool_calls(upstream_response):
        return "tool_calls"
    incomplete_reason = (
        upstream_response.get("incomplete_details", {}) or {}
    ).get("reason")
    if incomplete_reason == "max_output_tokens":
        return "length"
    if incomplete_reason == "content_filter":
        return "content_filter"
    return "stop"


def resolve_chat_max_tokens(request: LegacyChatCompletionRequest) -> int | None:
    if request.max_tokens is not None:
        return request.max_tokens
    return request.max_completion_tokens


def extract_usage(upstream_response: dict[str, Any]) -> CompletionUsage:
    usage = upstream_response.get("usage", {}) or {}
    return CompletionUsage(
        prompt_tokens=int(usage.get("input_tokens", 0)),
        completion_tokens=int(usage.get("output_tokens", 0)),
        total_tokens=int(usage.get("total_tokens", 0)),
    )


def apply_stop_sequences(
    text: str,
    stop: str | list[str] | None,
    default_finish_reason: str,
) -> tuple[str, str]:
    if not stop:
        return text, default_finish_reason
    stops = [stop] if isinstance(stop, str) else stop
    earliest_pos = None
    for stop_seq in stops:
        idx = text.find(stop_seq)
        if idx == -1:
            continue
        if earliest_pos is None or idx < earliest_pos:
            earliest_pos = idx
    if earliest_pos is None:
        return text, default_finish_reason
    return text[:earliest_pos], "stop"


def _validate_supported_params(request: LegacyCompletionRequest) -> None:
    for field_name in UNSUPPORTED_FIELDS:
        value = getattr(request, field_name)
        if value is not None and not _is_effectively_ignored_unsupported_value(
            field_name, value
        ):
            raise UnsupportedParameterError(
                f"'{field_name}' is not supported by this compatibility proxy."
            )


def _validate_supported_chat_params(request: LegacyChatCompletionRequest) -> None:
    for field_name in UNSUPPORTED_CHAT_FIELDS:
        value = getattr(request, field_name)
        if value is not None and not _is_effectively_ignored_unsupported_value(
            field_name, value
        ):
            raise UnsupportedParameterError(
                f"'{field_name}' is not supported by this compatibility proxy."
            )


def _is_effectively_ignored_unsupported_value(field_name: str, value: Any) -> bool:
    if field_name in {"frequency_penalty", "presence_penalty"}:
        return isinstance(value, (int, float)) and float(value) == 0.0

    if field_name == "logit_bias":
        return isinstance(value, dict) and not value

    if field_name == "response_format":
        if isinstance(value, dict):
            response_type = value.get("type")
            return not value or response_type in {None, "text"}
        return False

    if field_name == "best_of":
        return isinstance(value, int) and value == 1

    if field_name == "suffix":
        return isinstance(value, str) and not value

    return False


def _build_shared_sampling_params(
    target_model: str,
    max_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    top_logprobs: int | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if max_tokens is not None and not _requires_non_empty_instructions(target_model):
        params["max_output_tokens"] = max_tokens
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
    if top_logprobs is not None:
        params["top_logprobs"] = top_logprobs
    return params


def _text_type_for_role(role: str) -> str:
    if role in {"user", "system", "developer"}:
        return "input_text"
    return "output_text"


def _build_tool_output_item(raw_message: dict[str, Any]) -> dict[str, Any]:
    call_id = raw_message.get("tool_call_id")
    if not isinstance(call_id, str) or not call_id.strip():
        raise UnsupportedParameterError("tool message requires non-empty tool_call_id.")

    output = _extract_text_from_message_content(raw_message.get("content"))
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _extract_assistant_tool_call_items(raw_message: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    function_call = raw_message.get("function_call")
    if isinstance(function_call, dict):
        name = function_call.get("name")
        arguments = function_call.get("arguments")
        if isinstance(name, str) and name.strip():
            items.append(
                {
                    "type": "function_call",
                    "call_id": raw_message.get("tool_call_id") or f"call_{uuid.uuid4().hex}",
                    "name": name,
                    "arguments": arguments if isinstance(arguments, str) else "",
                }
            )

    tool_calls = raw_message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function_obj = call.get("function")
            if not isinstance(function_obj, dict):
                continue
            function_name = function_obj.get("name")
            if not isinstance(function_name, str) or not function_name.strip():
                continue

            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id.strip():
                call_id = f"call_{uuid.uuid4().hex}"

            arguments = function_obj.get("arguments")
            items.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": function_name,
                    "arguments": arguments if isinstance(arguments, str) else "",
                }
            )

    return items


def _attach_reasoning(payload: dict[str, Any], reasoning_effort: str | None) -> None:
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}


def _requires_non_empty_instructions(target_model: str) -> bool:
    return "codex" in target_model.lower()


def _extract_chat_instructions(request: LegacyChatCompletionRequest) -> str | None:
    chunks: list[str] = []
    for message in request.messages:
        role = message.role.lower()
        if role not in {"system", "developer"}:
            continue
        text = _extract_text_from_message_content(message.content)
        if text:
            chunks.append(text)

    if not chunks:
        return None
    return "\n\n".join(chunks).strip() or None


def _extract_text_from_message_content(content: str | list[dict[str, Any]] | None) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if isinstance(part.get("text"), str):
            text = part["text"].strip()
            if text:
                parts.append(text)
            continue
        if isinstance(part.get("refusal"), str):
            refusal = part["refusal"].strip()
            if refusal:
                parts.append(refusal)
    return "\n".join(parts)


def _convert_chat_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise UnsupportedParameterError("tools[] entries must be objects.")

        tool_type = tool.get("type")
        if tool_type != "function":
            converted.append(tool)
            continue

        function_obj = tool.get("function")
        function_name = tool.get("name")
        if not isinstance(function_name, str) or not function_name.strip():
            if isinstance(function_obj, dict):
                candidate = function_obj.get("name")
                if isinstance(candidate, str) and candidate.strip():
                    function_name = candidate
        if not isinstance(function_name, str) or not function_name.strip():
            raise UnsupportedParameterError("tools[].function.name is required for function tools.")

        normalized: dict[str, Any] = {"type": "function", "name": function_name}
        description = tool.get("description")
        parameters = tool.get("parameters")
        strict = tool.get("strict")

        if isinstance(function_obj, dict):
            if isinstance(function_obj.get("description"), str):
                description = function_obj["description"]
            if isinstance(function_obj.get("parameters"), dict):
                parameters = function_obj["parameters"]
            if isinstance(function_obj.get("strict"), bool):
                strict = function_obj["strict"]

        if isinstance(description, str):
            normalized["description"] = description
        if isinstance(parameters, dict):
            normalized["parameters"] = parameters
        if isinstance(strict, bool):
            normalized["strict"] = strict

        converted.append(normalized)
    return converted


def _convert_chat_tool_choice(tool_choice: str | dict[str, Any]) -> str | dict[str, Any]:
    if isinstance(tool_choice, str):
        return tool_choice

    tool_type = tool_choice.get("type")
    if tool_type != "function":
        return tool_choice

    function_name = tool_choice.get("name")
    function_obj = tool_choice.get("function")
    if not isinstance(function_name, str) or not function_name.strip():
        if isinstance(function_obj, dict):
            candidate = function_obj.get("name")
            if isinstance(candidate, str) and candidate.strip():
                function_name = candidate
    if not isinstance(function_name, str) or not function_name.strip():
        raise UnsupportedParameterError("tool_choice.function.name is required when tool_choice.type='function'.")

    return {"type": "function", "name": function_name}
