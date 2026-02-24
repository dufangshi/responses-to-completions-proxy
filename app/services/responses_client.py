from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx

from app.config import Settings


class UpstreamAPIError(Exception):
    def __init__(self, status_code: int, payload: Any):
        super().__init__(f"Upstream API error {status_code}")
        self.status_code = status_code
        self.payload = payload


class BaseResponsesGateway(ABC):
    @abstractmethod
    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def stream_response(self, payload: dict[str, Any]) -> Any:
        raise NotImplementedError


class OpenAIResponsesGateway(BaseResponsesGateway):
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.upstream_base_url,
            timeout=settings.upstream_timeout_seconds,
            headers={
                "Authorization": f"Bearer {settings.upstream_api_key}",
                "Content-Type": "application/json",
            },
        )

    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._client.post("/responses", json=payload)
        if response.status_code >= 400:
            raise UpstreamAPIError(response.status_code, _safe_json(response))
        return response.json()

    async def stream_response(self, payload: dict[str, Any]) -> Any:
        raise NotImplementedError("Streaming bridge not implemented yet.")

    async def close(self) -> None:
        await self._client.aclose()


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"error": {"message": response.text}}
