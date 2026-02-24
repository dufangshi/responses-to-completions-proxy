from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import Settings
from app.services.raw_io_logger import RawIOLogger


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
    async def stream_response(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        raise NotImplementedError


class OpenAIResponsesGateway(BaseResponsesGateway):
    def __init__(self, settings: Settings, raw_logger: RawIOLogger | None = None):
        self._settings = settings
        self._raw_logger = raw_logger or RawIOLogger.from_settings(settings)
        self._client = httpx.AsyncClient(
            base_url=settings.upstream_base_url,
            timeout=settings.upstream_timeout_seconds,
            headers={
                "Authorization": f"Bearer {settings.upstream_api_key}",
                "Content-Type": "application/json",
            },
        )

    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._raw_logger.log(
            "upstream.request",
            {"path": "/responses", "stream": False, "payload": payload},
        )
        response = await self._client.post("/responses", json=payload)
        self._raw_logger.log(
            "upstream.response",
            {
                "path": "/responses",
                "stream": False,
                "status_code": response.status_code,
                "body": response.text,
            },
        )
        if response.status_code >= 400:
            raise UpstreamAPIError(response.status_code, _safe_json(response))
        try:
            return response.json()
        except ValueError:
            raise UpstreamAPIError(
                502,
                {
                    "error": {
                        "message": "Upstream returned non-JSON success response.",
                        "type": "upstream_invalid_response",
                        "code": "invalid_upstream_response",
                        "upstream_status": response.status_code,
                        "body_preview": response.text[:2000],
                    }
                },
            ) from None

    async def stream_response(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        self._raw_logger.log(
            "upstream.request",
            {"path": "/responses", "stream": True, "payload": payload},
        )
        headers = {"Accept": "text/event-stream"}
        request = self._client.build_request(
            "POST",
            "/responses",
            json=payload,
            headers=headers,
        )
        response = await self._client.send(request, stream=True)
        self._raw_logger.log(
            "upstream.response",
            {
                "path": "/responses",
                "stream": True,
                "status_code": response.status_code,
            },
        )
        if response.status_code >= 400:
            body = await response.aread()
            self._raw_logger.log(
                "upstream.response.error_body",
                {
                    "path": "/responses",
                    "stream": True,
                    "status_code": response.status_code,
                    "body": body.decode("utf-8", errors="replace"),
                },
            )
            await response.aclose()
            raise UpstreamAPIError(response.status_code, _safe_json(body))

        async def iterator() -> AsyncIterator[str]:
            try:
                async for line in response.aiter_lines():
                    self._raw_logger.log(
                        "upstream.response.stream_line",
                        {"path": "/responses", "line": line},
                    )
                    yield line
            finally:
                await response.aclose()

        return iterator()

    async def close(self) -> None:
        await self._client.aclose()


def _safe_json(response: httpx.Response | bytes) -> Any:
    if isinstance(response, bytes):
        try:
            return httpx.Response(500, content=response).json()
        except ValueError:
            return {"error": {"message": response.decode("utf-8", errors="replace")}}
    try:
        return response.json()
    except ValueError:
        return {"error": {"message": response.text}}
