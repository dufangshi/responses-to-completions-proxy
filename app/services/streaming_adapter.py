from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from app.models.legacy_completions import CompletionUsage


def encode_sse_json(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def encode_sse_done() -> bytes:
    return b"data: [DONE]\n\n"


def wants_usage_chunk(stream_options: dict[str, Any] | None) -> bool:
    return bool(stream_options and stream_options.get("include_usage"))


async def iter_upstream_sse_events(
    upstream_lines: AsyncIterator[str],
) -> AsyncIterator[dict[str, Any] | str]:
    event_name: str | None = None
    data_lines: list[str] = []

    async for raw_line in upstream_lines:
        line = raw_line.rstrip("\r")
        if line == "":
            payload = _flush_event(event_name, data_lines)
            if payload is not None:
                yield payload
            event_name = None
            data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue

    payload = _flush_event(event_name, data_lines)
    if payload is not None:
        yield payload


def fallback_stream_identity(prefix: str) -> tuple[str, int]:
    return f"{prefix}_{uuid.uuid4().hex}", int(time.time())


def completion_stream_chunk(
    response_id: str,
    created: int,
    model: str,
    text: str,
    index: int = 0,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "text": text,
                "index": index,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }


def completion_stream_usage_chunk(
    response_id: str,
    created: int,
    model: str,
    usage: CompletionUsage,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [],
        "usage": usage.model_dump(),
    }


def chat_stream_chunk(
    response_id: str,
    created: int,
    model: str,
    delta: dict[str, Any] | None,
    index: int = 0,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": index,
                "delta": delta or {},
                "finish_reason": finish_reason,
            }
        ],
    }


def chat_stream_usage_chunk(
    response_id: str,
    created: int,
    model: str,
    usage: CompletionUsage,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": usage.model_dump(),
    }


def _flush_event(event_name: str | None, data_lines: list[str]) -> dict[str, Any] | str | None:
    if not data_lines:
        return None
    merged = "\n".join(data_lines)
    if merged == "[DONE]":
        return "[DONE]"
    try:
        event_payload = json.loads(merged)
    except json.JSONDecodeError:
        return {"event": event_name, "data": merged}
    if isinstance(event_payload, dict):
        if event_name and "event" not in event_payload:
            event_payload["event"] = event_name
        return event_payload
    return {"event": event_name, "data": event_payload}
