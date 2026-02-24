from __future__ import annotations

from contextvars import ContextVar, Token

_CURRENT_REQUEST_ID: ContextVar[str | None] = ContextVar("current_request_id", default=None)


def set_current_request_id(request_id: str) -> Token[str | None]:
    return _CURRENT_REQUEST_ID.set(request_id)


def reset_current_request_id(token: Token[str | None]) -> None:
    _CURRENT_REQUEST_ID.reset(token)


def get_current_request_id() -> str | None:
    return _CURRENT_REQUEST_ID.get()
