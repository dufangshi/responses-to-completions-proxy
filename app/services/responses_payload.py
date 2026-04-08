from __future__ import annotations

from typing import Any


_MAX_INPUT_NAME_LENGTH = 128


class ResponsesValidationError(ValueError):
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


def normalize_responses_input(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    raw_input = normalized.get("input")

    if isinstance(raw_input, str):
        normalized["input"] = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": raw_input}],
            }
        ]
        return normalized

    if isinstance(raw_input, list) and raw_input and all(isinstance(item, str) for item in raw_input):
        normalized["input"] = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": item}],
            }
            for item in raw_input
        ]
        return normalized

    validate_responses_input_items(raw_input)
    return normalized


def validate_responses_input_items(raw_input: Any) -> None:
    if not isinstance(raw_input, list):
        return

    for index, item in enumerate(raw_input):
        if not isinstance(item, dict):
            continue

        if "name" not in item:
            continue

        param = f"input[{index}].name"
        name = item.get("name")
        if not isinstance(name, str):
            raise ResponsesValidationError(
                f"{param} must be a string.",
                param=param,
                code="invalid_type",
            )

        stripped = name.strip()
        if not stripped:
            raise ResponsesValidationError(
                f"{param} must be a non-empty string.",
                param=param,
                code="invalid_value",
            )

        if len(stripped) > _MAX_INPUT_NAME_LENGTH:
            raise ResponsesValidationError(
                f"{param} must be at most {_MAX_INPUT_NAME_LENGTH} characters.",
                param=param,
                code="string_above_max_length",
            )

        if stripped != name:
            item["name"] = stripped
