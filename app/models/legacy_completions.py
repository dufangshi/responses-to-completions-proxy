from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LegacyCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    prompt: str | list[str]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int = 1
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    logprobs: int | None = None
    echo: bool = False
    stop: str | list[str] | None = None
    user: str | None = None
    suffix: str | None = None
    best_of: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    seed: int | None = None

    @field_validator("n")
    @classmethod
    def validate_n(cls, value: int) -> int:
        if value < 1:
            raise ValueError("n must be >= 1")
        return value

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, list) and len(value) == 0:
            raise ValueError("prompt list cannot be empty")
        return value


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Any = None
    finish_reason: str | None = None


class CompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LegacyCompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice] = Field(default_factory=list)
    usage: CompletionUsage = Field(default_factory=CompletionUsage)
