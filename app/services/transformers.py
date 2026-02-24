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


class UnsupportedParameterError(ValueError):
    pass


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
    "tools",
    "tool_choice",
)


def build_responses_payload(
    request: LegacyCompletionRequest,
    target_model: str,
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
    if request.max_tokens is not None:
        payload["max_output_tokens"] = request.max_tokens
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.user:
        payload["user"] = request.user
    if request.logprobs is not None:
        payload["top_logprobs"] = request.logprobs
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
) -> dict[str, Any]:
    _validate_supported_chat_params(request)
    return {
        "model": target_model,
        "input": [build_chat_input_message(message.model_dump()) for message in request.messages],
        "store": False,
        **_build_shared_sampling_params(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            user=request.user,
            top_logprobs=request.top_logprobs,
        ),
    }


def build_legacy_chat_completion_response(
    request: LegacyChatCompletionRequest,
    upstream_results: list[dict[str, Any]],
    response_model_name: str,
) -> LegacyChatCompletionResponse:
    choices: list[ChatCompletionChoice] = []
    usage = ChatCompletionUsage()
    for index, result in enumerate(upstream_results):
        output_text = extract_output_text(result)
        output_text, finish_reason = apply_stop_sequences(
            text=output_text,
            stop=request.stop,
            default_finish_reason=map_finish_reason(result),
        )
        choices.append(
            ChatCompletionChoice(
                index=index,
                message=ChatCompletionMessageOut(content=output_text),
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


def build_chat_input_message(raw_message: dict[str, Any]) -> dict[str, Any]:
    role = raw_message.get("role")
    if not isinstance(role, str):
        raise UnsupportedParameterError("messages[].role must be a string.")

    content = raw_message.get("content")
    if isinstance(content, str):
        content_items = [{"type": "input_text", "text": content}]
    elif isinstance(content, list):
        content_items = [convert_chat_content_part(part) for part in content]
    else:
        raise UnsupportedParameterError("messages[].content must be string or content parts array.")

    built: dict[str, Any] = {"role": role, "content": content_items}
    if isinstance(raw_message.get("name"), str):
        built["name"] = raw_message["name"]
    return built


def convert_chat_content_part(part: Any) -> dict[str, Any]:
    if not isinstance(part, dict):
        raise UnsupportedParameterError("messages[].content[] entries must be objects.")

    part_type = part.get("type")
    if part_type == "text":
        text = part.get("text")
        if not isinstance(text, str):
            raise UnsupportedParameterError("messages[].content[].text must be a string.")
        return {"type": "input_text", "text": text}

    if part_type == "input_text":
        text = part.get("text")
        if not isinstance(text, str):
            raise UnsupportedParameterError("messages[].content[].text must be a string.")
        return {"type": "input_text", "text": text}

    if part_type == "image_url":
        image_field = part.get("image_url")
        image_url = image_field.get("url") if isinstance(image_field, dict) else image_field
        if not isinstance(image_url, str):
            raise UnsupportedParameterError("messages[].content[].image_url must contain a url string.")
        return {"type": "input_image", "image_url": image_url}

    if part_type == "input_image":
        image_url = part.get("image_url")
        if not isinstance(image_url, str):
            raise UnsupportedParameterError("messages[].content[].image_url must be a string.")
        built = {"type": "input_image", "image_url": image_url}
        if "detail" in part:
            built["detail"] = part["detail"]
        return built

    raise UnsupportedParameterError(
        f"messages[].content[] type '{part_type}' is not supported by this compatibility proxy."
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


def map_finish_reason(upstream_response: dict[str, Any]) -> str:
    incomplete_reason = (
        upstream_response.get("incomplete_details", {}) or {}
    ).get("reason")
    if incomplete_reason == "max_output_tokens":
        return "length"
    if incomplete_reason == "content_filter":
        return "content_filter"
    return "stop"


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
    if request.stream:
        raise UnsupportedParameterError("stream=true is not implemented yet.")
    for field_name in UNSUPPORTED_FIELDS:
        if getattr(request, field_name) is not None:
            raise UnsupportedParameterError(
                f"'{field_name}' is not supported by this compatibility proxy."
            )


def _validate_supported_chat_params(request: LegacyChatCompletionRequest) -> None:
    if request.stream:
        raise UnsupportedParameterError("stream=true is not implemented yet.")
    for field_name in UNSUPPORTED_CHAT_FIELDS:
        if getattr(request, field_name) is not None:
            raise UnsupportedParameterError(
                f"'{field_name}' is not supported by this compatibility proxy."
            )


def _build_shared_sampling_params(
    max_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    user: str | None,
    top_logprobs: int | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if max_tokens is not None:
        params["max_output_tokens"] = max_tokens
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
    if user:
        params["user"] = user
    if top_logprobs is not None:
        params["top_logprobs"] = top_logprobs
    return params
