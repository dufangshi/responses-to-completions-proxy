from __future__ import annotations

from typing import NamedTuple


class ModelLimits(NamedTuple):
    context_window: int
    max_output_tokens: int


KNOWN_MODEL_LIMITS: dict[str, ModelLimits] = {
    "gpt-5.3-codex": ModelLimits(context_window=400_000, max_output_tokens=128_000),
    "gemini-3-flash-preview": ModelLimits(context_window=1_048_576, max_output_tokens=65_536),
    "gemini-3-pro-preview": ModelLimits(context_window=1_048_576, max_output_tokens=65_536),
    "gemini-3.1-pro-preview": ModelLimits(context_window=1_048_576, max_output_tokens=65_536),
}


DEFAULT_MODEL_LIMITS = ModelLimits(context_window=200_000, max_output_tokens=8_192)


def resolve_model_limits(model_id: str) -> ModelLimits:
    normalized = model_id.strip().lower()
    if normalized in KNOWN_MODEL_LIMITS:
        return KNOWN_MODEL_LIMITS[normalized]
    return DEFAULT_MODEL_LIMITS


def resolve_model_max_output_tokens(model_id: str) -> int | None:
    normalized = model_id.strip().lower()
    limits = KNOWN_MODEL_LIMITS.get(normalized)
    if limits is None:
        return None
    return limits.max_output_tokens
