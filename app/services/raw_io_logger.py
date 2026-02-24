from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from app.config import Settings
from app.services.request_context import get_current_request_id


class RawIOLogger:
    def __init__(self, enabled: bool, path: str, max_chars: int, keep_requests: int):
        self._enabled = enabled
        self._path = Path(path)
        self._max_chars = max_chars
        self._keep_requests = keep_requests
        self._lock = threading.Lock()

        if self._enabled:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_settings(cls, settings: Settings) -> "RawIOLogger":
        return cls(
            enabled=settings.raw_io_log_enabled,
            path=settings.raw_io_log_path,
            max_chars=settings.raw_io_log_max_chars,
            keep_requests=settings.raw_io_log_keep_requests,
        )

    def log(self, kind: str, payload: dict[str, Any]) -> None:
        if not self._enabled:
            return

        request_id = get_current_request_id()
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "kind": kind,
            "request_id": payload.get("request_id") or request_id,
            **self._truncate_dict(payload),
        }
        line = json.dumps(record, ensure_ascii=False, default=str)

        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
            if kind == "proxy.response":
                self._prune_to_recent_requests_locked()

    def _truncate_value(self, value: Any) -> Any:
        if isinstance(value, str):
            if len(value) <= self._max_chars:
                return value
            return f"{value[:self._max_chars]}...[TRUNCATED {len(value) - self._max_chars} chars]"
        if isinstance(value, list):
            return [self._truncate_value(item) for item in value]
        if isinstance(value, dict):
            return self._truncate_dict(value)
        return value

    def _truncate_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        return {key: self._truncate_value(value) for key, value in data.items()}

    def _prune_to_recent_requests_locked(self) -> None:
        if self._keep_requests <= 0 or not self._path.exists():
            return

        lines = self._path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return

        parsed_records: list[tuple[str, dict[str, Any] | None]] = []
        request_order: list[str] = []
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                parsed_records.append((line, None))
                continue

            parsed_records.append((line, record))
            if not isinstance(record, dict):
                continue
            if record.get("kind") != "proxy.request":
                continue
            request_id = record.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                continue
            request_order.append(request_id)

        if len(request_order) <= self._keep_requests:
            return

        recent_ids = set(request_order[-self._keep_requests :])
        kept_lines: list[str] = []
        for line, record in parsed_records:
            if record is None:
                continue
            request_id = record.get("request_id")
            if isinstance(request_id, str) and request_id in recent_ids:
                kept_lines.append(line)

        self._path.write_text(
            ("\n".join(kept_lines) + "\n") if kept_lines else "",
            encoding="utf-8",
        )
