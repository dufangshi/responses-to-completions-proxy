from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

ALLOWED_REASONING_EFFORTS = {"low", "medium", "high", "xhigh"}
REASONING_EFFORT_MODELS = {
    "gpt-5.3-codex",
    "gpt-5.4",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
}
ALLOWED_UPSTREAM_MODES = {"responses", "messages"}


def _parse_reasoning_effort(raw_value: str) -> str | None:
    value = raw_value.strip().lower()
    if not value:
        return None
    if value not in ALLOWED_REASONING_EFFORTS:
        raise ValueError(
            "DEFAULT_REASONING_EFFORT must be one of: low, medium, high, xhigh."
        )
    return value


def _parse_upstream_mode(raw_value: str) -> str:
    value = raw_value.strip().lower()
    if not value:
        return "responses"
    if value not in ALLOWED_UPSTREAM_MODES:
        raise ValueError(
            "UPSTREAM_MODE must be one of: responses, messages."
        )
    return value


def _supports_reasoning_effort(model_name: str) -> bool:
    return model_name.strip().lower() in REASONING_EFFORT_MODELS


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


def _parse_non_negative_float(raw_value: str, default: float) -> float:
    value = raw_value.strip()
    if not value:
        return default
    parsed = float(value)
    if parsed < 0:
        raise ValueError("UPSTREAM_MIN_REQUEST_INTERVAL_SECONDS must be >= 0.")
    return parsed


def _parse_csv(raw_value: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = raw_value.strip()
    if not value:
        return default
    parts = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not parts:
        return default
    return parts


def _parse_optional_str(raw_value: str) -> str | None:
    value = raw_value.strip()
    if not value:
        return None
    return value


def _ensure_url_suffix(base_url: str, suffix: str) -> str:
    normalized_base = base_url.strip().rstrip("/")
    normalized_suffix = suffix.strip()
    if not normalized_suffix:
        return normalized_base
    if not normalized_suffix.startswith("/"):
        normalized_suffix = f"/{normalized_suffix}"
    if normalized_base.endswith(normalized_suffix):
        return normalized_base
    return f"{normalized_base}{normalized_suffix}"


def _derive_upstream_root(raw_base_url: str) -> str:
    normalized = raw_base_url.strip().rstrip("/")
    if not normalized:
        return normalized
    for suffix in ("/v1", "/v1beta", "/antigravity"):
        if normalized.endswith(suffix):
            trimmed = normalized[: -len(suffix)].rstrip("/")
            if trimmed:
                return trimmed
    return normalized


def _parse_force_model_list(raw_value: str) -> tuple[str, ...]:
    value = raw_value.strip()
    if not value:
        return ()

    if value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                models = tuple(str(item).strip() for item in parsed if str(item).strip())
                if models:
                    return models
        except json.JSONDecodeError:
            inner = value.strip("[]").strip()
            if not inner:
                return ()
            return tuple(
                part.strip().strip("\"'") for part in inner.split(",") if part.strip().strip("\"'")
            )

    return tuple(part.strip() for part in value.split(",") if part.strip())


@dataclass(slots=True)
class Settings:
    app_host: str
    app_port: int
    upstream_base_url: str
    upstream_api_key: str
    upstream_mode: str
    upstream_streaming_enabled: bool
    upstream_messages_api_version: str
    upstream_timeout_seconds: float
    upstream_min_request_interval_seconds: float
    upstream_fallback_model: str | None
    use_force_model: bool
    force_upstream_models: tuple[str, ...]
    default_upstream_model: str
    default_reasoning_effort: str | None
    default_upstream_speed: str | None
    raw_io_log_enabled: bool
    raw_io_log_path: str
    raw_io_log_max_chars: int
    raw_io_log_keep_requests: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        upstream_mode = _parse_upstream_mode(os.getenv("UPSTREAM_MODE", "responses"))
        upstream_root = _derive_upstream_root(
            os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com")
        )
        upstream_base_url = _ensure_url_suffix(
            os.getenv("UPSTREAM_MODE_BASE_URL", "").strip() or upstream_root,
            "/v1",
        )
        upstream_api_key = os.getenv("UPSTREAM_API_KEY", "").strip()
        if not upstream_api_key:
            raise ValueError("UPSTREAM_API_KEY is required.")

        upstream_interval_raw = os.getenv(
            "UPSTREAM_MIN_REQUEST_INTERVAL_SECONDS", "10"
        ).strip()
        upstream_fallback_raw = os.getenv(
            "UPSTREAM_FALLBACK_MODEL", ""
        ).strip()
        return cls(
            app_host=os.getenv("APP_HOST", "127.0.0.1"),
            app_port=int(os.getenv("APP_PORT", "18010")),
            upstream_base_url=upstream_base_url,
            upstream_api_key=upstream_api_key,
            upstream_mode=upstream_mode,
            upstream_streaming_enabled=_parse_bool(
                os.getenv("UPSTREAM_STREAMING_ENABLED", "true"),
                default=True,
            ),
            upstream_messages_api_version=os.getenv(
                "UPSTREAM_MESSAGES_API_VERSION", "2023-06-01"
            ).strip()
            or "2023-06-01",
            upstream_timeout_seconds=float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "120")),
            upstream_min_request_interval_seconds=_parse_non_negative_float(
                upstream_interval_raw,
                default=10.0,
            ),
            upstream_fallback_model=_parse_optional_str(
                upstream_fallback_raw
            ),
            use_force_model=_parse_bool(os.getenv("USE_FORCE_MODEL", ""), default=False),
            force_upstream_models=_parse_force_model_list(
                os.getenv("FORCE_UPSTREAM_MODEL", "")
            ),
            default_upstream_model=os.getenv("DEFAULT_UPSTREAM_MODEL", "gpt-5.4"),
            default_reasoning_effort=_parse_reasoning_effort(
                os.getenv("DEFAULT_REASONING_EFFORT", "medium")
            ),
            default_upstream_speed=_parse_optional_str(
                os.getenv("DEFAULT_UPSTREAM_SPEED", "fast")
            ),
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
        if self.use_force_model and self.force_upstream_models:
            return self.force_upstream_models[0]
        return self.default_upstream_model

    def resolve_model_and_reasoning(self, client_model: str | None) -> tuple[str, str | None]:
        resolved_model = self.resolve_model(client_model)
        return resolved_model, self.reasoning_effort_for_model(resolved_model)

    def reasoning_effort_for_model(self, model_name: str) -> str | None:
        if not _supports_reasoning_effort(model_name):
            return None
        return self.default_reasoning_effort

    def force_model_chain(self) -> tuple[str, ...]:
        if not self.use_force_model:
            return ()
        deduped: list[str] = []
        seen: set[str] = set()
        for model in self.force_upstream_models:
            normalized = model.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(model.strip())
        return tuple(deduped)
