from __future__ import annotations

import asyncio
import copy
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ResponsesSessionState:
    session_key: str
    response_id: str
    model: str
    instructions_hash: str | None
    tools_hash: str | None
    tool_choice_hash: str | None
    full_input: list[dict[str, Any]]
    match_input: list[dict[str, Any]] | None = None
    source_fingerprint: str | None = None
    root_fingerprint: str | None = None
    updated_at: float = 0.0


class ResponsesSessionStore:
    def __init__(self, max_entries: int = 256):
        self._max_entries = max(1, max_entries)
        self._items: OrderedDict[str, ResponsesSessionState] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, session_key: str) -> ResponsesSessionState | None:
        async with self._lock:
            state = self._items.get(session_key)
            if state is None:
                return None
            self._items.move_to_end(session_key)
            return copy.deepcopy(state)

    async def put(self, state: ResponsesSessionState) -> None:
        async with self._lock:
            state = copy.deepcopy(state)
            state.updated_at = time.time()
            self._items[state.session_key] = state
            self._items.move_to_end(state.session_key)
            while len(self._items) > self._max_entries:
                self._items.popitem(last=False)

    async def delete(self, session_key: str) -> None:
        async with self._lock:
            self._items.pop(session_key, None)

    async def list_states(self) -> list[ResponsesSessionState]:
        async with self._lock:
            items = list(self._items.values())
            items.reverse()
            return [copy.deepcopy(item) for item in items]
