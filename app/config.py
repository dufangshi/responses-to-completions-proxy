from __future__ import annotations

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _parse_model_map(raw_value: str) -> dict[str, str]:
    raw_value = raw_value.strip()
    if not raw_value:
        return {}

    if raw_value.startswith("{"):
        parsed = json.loads(raw_value)
        if not isinstance(parsed, dict):
            raise ValueError("MODEL_MAP JSON must be an object.")
        return {str(k): str(v) for k, v in parsed.items()}

    model_map: dict[str, str] = {}
    for segment in raw_value.split(","):
        part = segment.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                "MODEL_MAP must be JSON or comma-separated legacy:target pairs."
            )
        legacy_model, target_model = part.split(":", 1)
        model_map[legacy_model.strip()] = target_model.strip()
    return model_map


def _parse_reasoning_effort(raw_value: str) -> str | None:
    value = raw_value.strip().lower()
    if not value:
        return None
    allowed_values = {"low", "medium", "high", "xhigh"}
    if value not in allowed_values:
        raise ValueError(
            "DEFAULT_REASONING_EFFORT must be one of: low, medium, high, xhigh."
        )
    return value


def _parse_bool(raw_value: str, default: bool = False) -> bool:
    value = raw_value.strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _parse_positive_int(raw_value: str, default: int) -> int:
    value = raw_value.strip()
    if not value:
        return default
    parsed = int(value)
    if parsed < 1:
        raise ValueError("RAW_IO_LOG_MAX_CHARS must be >= 1.")
    return parsed


def _parse_non_negative_int(raw_value: str, default: int) -> int:
    value = raw_value.strip()
    if not value:
        return default
    parsed = int(value)
    if parsed < 0:
        raise ValueError("RAW_IO_LOG_KEEP_REQUESTS must be >= 0.")
    return parsed


@dataclass(slots=True)
class Settings:
    app_host: str
    app_port: int
    upstream_base_url: str
    upstream_api_key: str
    upstream_timeout_seconds: float
    default_upstream_model: str
    default_reasoning_effort: str | None
    model_map: dict[str, str]
    raw_io_log_enabled: bool
    raw_io_log_path: str
    raw_io_log_max_chars: int
    raw_io_log_keep_requests: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        raw_model_map = os.getenv("MODEL_MAP", "")
        model_map = _parse_model_map(raw_model_map)
        return cls(
            app_host=os.getenv("APP_HOST", "127.0.0.1"),
            app_port=int(os.getenv("APP_PORT", "18010")),
            upstream_base_url=os.getenv(
                "UPSTREAM_BASE_URL", "https://api.openai.com/v1"
            ).rstrip("/"),
            upstream_api_key=os.getenv("UPSTREAM_API_KEY", ""),
            upstream_timeout_seconds=float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "120")),
            default_upstream_model=os.getenv("DEFAULT_UPSTREAM_MODEL", "gpt-5.3-codex"),
            default_reasoning_effort=_parse_reasoning_effort(
                os.getenv("DEFAULT_REASONING_EFFORT", "high")
            ),
            model_map=model_map,
            raw_io_log_enabled=_parse_bool(os.getenv("RAW_IO_LOG_ENABLED", "")),
            raw_io_log_path=os.getenv("RAW_IO_LOG_PATH", "logs/raw_io.jsonl"),
            raw_io_log_max_chars=_parse_positive_int(
                os.getenv("RAW_IO_LOG_MAX_CHARS", "120000"),
                default=120000,
            ),
            raw_io_log_keep_requests=_parse_non_negative_int(
                os.getenv("RAW_IO_LOG_KEEP_REQUESTS", "10"),
                default=10,
            ),
        )

    def resolve_model(self, client_model: str | None) -> str:
        if client_model and client_model in self.model_map:
            return self.model_map[client_model]
        if client_model:
            return client_model
        return self.default_upstream_model
