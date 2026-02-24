from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatCompletionMessageIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict[str, Any]]
    name: str | None = None
    tool_call_id: str | None = None


class LegacyChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatCompletionMessageIn]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    user: str | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    seed: int | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, value: list[ChatCompletionMessageIn]) -> list[ChatCompletionMessageIn]:
        if not value:
            raise ValueError("messages cannot be empty")
        return value

    @field_validator("n")
    @classmethod
    def validate_n(cls, value: int) -> int:
        if value < 1:
            raise ValueError("n must be >= 1")
        return value


class ChatCompletionMessageOut(BaseModel):
    role: str = "assistant"
    content: str
    refusal: str | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessageOut
    logprobs: Any = None
    finish_reason: str | None = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LegacyChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice] = Field(default_factory=list)
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)
