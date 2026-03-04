from __future__ import annotations

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv

ALLOWED_REASONING_EFFORTS = {"low", "medium", "high", "xhigh"}
REASONING_EFFORT_MODEL = "gpt-5.3-codex"


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
    if value not in ALLOWED_REASONING_EFFORTS:
        raise ValueError(
            "DEFAULT_REASONING_EFFORT must be one of: low, medium, high, xhigh."
        )
    return value


def _extract_model_reasoning_effort(model_name: str | None) -> tuple[str | None, str | None]:
    if not isinstance(model_name, str):
        return model_name, None

    raw = model_name.strip()
    if not raw:
        return None, None

    lowered = raw.lower()
    for effort in ALLOWED_REASONING_EFFORTS:
        suffix = f":{effort}"
        if lowered.endswith(suffix):
            base = raw[: -len(suffix)].strip()
            if base:
                return base, effort
            return None, effort
    return raw, None


def _supports_reasoning_effort(model_name: str) -> bool:
    return model_name.strip().lower() == REASONING_EFFORT_MODEL


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
        raise ValueError("ANTIGRAVITY_MIN_REQUEST_INTERVAL_SECONDS must be >= 0.")
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
    upstream_gemini_base_url: str
    upstream_gemini_api_key: str
    upstream_timeout_seconds: float
    gemini_min_request_interval_seconds: float
    gemini_fallback_model: str | None
    use_force_model: bool
    force_upstream_models: tuple[str, ...]
    default_upstream_model: str
    default_reasoning_effort: str | None
    model_map: dict[str, str]
    openai_model_prefixes: tuple[str, ...]
    raw_io_log_enabled: bool
    raw_io_log_path: str
    raw_io_log_max_chars: int
    raw_io_log_keep_requests: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        raw_model_map = os.getenv("MODEL_MAP", "")
        model_map = _parse_model_map(raw_model_map)
        upstream_root = _derive_upstream_root(
            os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com")
        )
        openai_base_url = _ensure_url_suffix(
            os.getenv("UPSTREAM_OPENAI_BASE_URL", "").strip() or upstream_root,
            "/v1",
        )
        antigravity_base_url = _ensure_url_suffix(
            os.getenv("UPSTREAM_ANTIGRAVITY_BASE_URL", "").strip() or upstream_root,
            "/antigravity",
        )
        openai_api_key = os.getenv("UPSTREAM_OPENAI_API_KEY", "").strip()
        if not openai_api_key:
            raise ValueError("UPSTREAM_OPENAI_API_KEY is required.")

        antigravity_api_key = os.getenv("UPSTREAM_ANTIGRAVITY_API_KEY", "").strip()
        if not antigravity_api_key:
            raise ValueError("UPSTREAM_ANTIGRAVITY_API_KEY is required.")

        antigravity_interval_raw = os.getenv(
            "ANTIGRAVITY_MIN_REQUEST_INTERVAL_SECONDS", "10"
        ).strip()
        antigravity_fallback_raw = os.getenv(
            "ANTIGRAVITY_FALLBACK_MODEL", "gemini-3-flash-preview"
        ).strip()
        return cls(
            app_host=os.getenv("APP_HOST", "127.0.0.1"),
            app_port=int(os.getenv("APP_PORT", "18010")),
            upstream_base_url=openai_base_url,
            upstream_api_key=openai_api_key,
            upstream_gemini_base_url=antigravity_base_url,
            upstream_gemini_api_key=antigravity_api_key,
            upstream_timeout_seconds=float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "120")),
            gemini_min_request_interval_seconds=_parse_non_negative_float(
                antigravity_interval_raw,
                default=10.0,
            ),
            gemini_fallback_model=_parse_optional_str(
                antigravity_fallback_raw
            ),
            use_force_model=_parse_bool(os.getenv("USE_FORCE_MODEL", ""), default=False),
            force_upstream_models=_parse_force_model_list(
                os.getenv("FORCE_UPSTREAM_MODEL", "")
            ),
            default_upstream_model=os.getenv("DEFAULT_UPSTREAM_MODEL", "gpt-5.3-codex"),
            default_reasoning_effort=_parse_reasoning_effort(
                os.getenv("DEFAULT_REASONING_EFFORT", "high")
            ),
            model_map=model_map,
            openai_model_prefixes=_parse_csv(
                os.getenv(
                    "OPENAI_MODEL_PREFIXES",
                    "gpt-,o1,o3,o4,text-embedding,text-moderation,whisper,tts,dall-e,omni",
                ),
                default=(
                    "gpt-",
                    "o1",
                    "o3",
                    "o4",
                    "text-embedding",
                    "text-moderation",
                    "whisper",
                    "tts",
                    "dall-e",
                    "omni",
                ),
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
        if client_model and client_model in self.model_map:
            return self.model_map[client_model]
        if client_model:
            return client_model
        return self.default_upstream_model

    def resolve_model_and_reasoning(self, client_model: str | None) -> tuple[str, str | None]:
        model_name, inline_reasoning = _extract_model_reasoning_effort(client_model)
        resolved_model = self.resolve_model(model_name)

        reasoning_effort = inline_reasoning
        if reasoning_effort is None:
            reasoning_effort = self.default_reasoning_effort

        if not _supports_reasoning_effort(resolved_model):
            reasoning_effort = None

        return resolved_model, reasoning_effort

    def is_openai_model(self, model_name: str) -> bool:
        normalized = model_name.strip().lower()
        if not normalized:
            return True
        return any(normalized.startswith(prefix) for prefix in self.openai_model_prefixes)

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
