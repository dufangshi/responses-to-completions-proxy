from __future__ import annotations

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from app.config import Settings
from app.services.gemini_adapter import (
    GeminiAdapterError,
    build_gemini_request_from_responses,
    build_openai_response_from_stream_state,
    extract_stream_delta_text,
    extract_text_from_candidate,
    extract_tool_calls_from_candidate,
    first_candidate,
    gemini_error_to_openai_error,
    gemini_response_to_openai_response,
)
from app.services.model_limits import resolve_model_max_output_tokens
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

    @abstractmethod
    async def close(self) -> None:
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
            {"provider": "openai", "path": "/responses", "stream": False, "payload": payload},
        )
        response = await self._client.post("/responses", json=payload)
        self._raw_logger.log(
            "upstream.response",
            {
                "provider": "openai",
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
            {"provider": "openai", "path": "/responses", "stream": True, "payload": payload},
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
                "provider": "openai",
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
                    "provider": "openai",
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
                        {"provider": "openai", "path": "/responses", "line": line},
                    )
                    yield line
            finally:
                await response.aclose()

        return iterator()

    async def close(self) -> None:
        await self._client.aclose()


class GeminiResponsesGateway(BaseResponsesGateway):
    def __init__(self, settings: Settings, raw_logger: RawIOLogger | None = None):
        self._settings = settings
        self._raw_logger = raw_logger or RawIOLogger.from_settings(settings)
        self._client = httpx.AsyncClient(
            base_url=settings.upstream_gemini_base_url,
            timeout=settings.upstream_timeout_seconds,
            headers={
                "x-goog-api-key": settings.upstream_gemini_api_key,
                "Content-Type": "application/json",
            },
        )

    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        model = _extract_model(payload)
        normalized_payload = _normalize_max_output_tokens_for_model(payload, model)
        if not self._settings.upstream_gemini_api_key:
            raise UpstreamAPIError(
                500,
                {
                    "error": {
                        "message": "UPSTREAM_GEMINI_API_KEY is required for Gemini models.",
                        "type": "server_error",
                        "param": None,
                        "code": "gemini_api_key_missing",
                    }
                },
            )
        try:
            gemini_payload = build_gemini_request_from_responses(normalized_payload)
        except GeminiAdapterError as exc:
            raise UpstreamAPIError(
                400,
                {
                    "error": {
                        "message": str(exc),
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_gemini_request",
                    }
                },
            ) from exc

        path = f"/models/{model}:generateContent"
        self._raw_logger.log(
            "upstream.request",
            {
                "provider": "gemini",
                "path": path,
                "stream": False,
                "payload": gemini_payload,
            },
        )
        response = await self._post_with_retry(path=path, payload=gemini_payload)
        self._raw_logger.log(
            "upstream.response",
            {
                "provider": "gemini",
                "path": path,
                "stream": False,
                "status_code": response.status_code,
                "body": response.text,
            },
        )
        if response.status_code >= 400:
            raw_error = _safe_json(response)
            raise UpstreamAPIError(response.status_code, gemini_error_to_openai_error(response.status_code, raw_error))

        try:
            parsed = response.json()
        except ValueError:
            raise UpstreamAPIError(
                502,
                {
                    "error": {
                        "message": "Gemini upstream returned non-JSON success response.",
                        "type": "upstream_invalid_response",
                        "param": None,
                        "code": "invalid_upstream_response",
                    }
                },
            ) from None

        return gemini_response_to_openai_response(parsed, model=model)

    async def stream_response(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        model = _extract_model(payload)
        normalized_payload = _normalize_max_output_tokens_for_model(payload, model)
        if not self._settings.upstream_gemini_api_key:
            raise UpstreamAPIError(
                500,
                {
                    "error": {
                        "message": "UPSTREAM_GEMINI_API_KEY is required for Gemini models.",
                        "type": "server_error",
                        "param": None,
                        "code": "gemini_api_key_missing",
                    }
                },
            )
        try:
            gemini_payload = build_gemini_request_from_responses(normalized_payload)
        except GeminiAdapterError as exc:
            raise UpstreamAPIError(
                400,
                {
                    "error": {
                        "message": str(exc),
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_gemini_request",
                    }
                },
            ) from exc

        path = f"/models/{model}:streamGenerateContent?alt=sse"
        self._raw_logger.log(
            "upstream.request",
            {
                "provider": "gemini",
                "path": path,
                "stream": True,
                "payload": gemini_payload,
            },
        )
        request = self._client.build_request(
            "POST",
            path,
            json=gemini_payload,
            headers={"Accept": "text/event-stream"},
        )
        response = await self._send_with_retry(request)
        self._raw_logger.log(
            "upstream.response",
            {
                "provider": "gemini",
                "path": path,
                "stream": True,
                "status_code": response.status_code,
            },
        )
        if response.status_code >= 400:
            body = await response.aread()
            self._raw_logger.log(
                "upstream.response.error_body",
                {
                    "provider": "gemini",
                    "path": path,
                    "stream": True,
                    "status_code": response.status_code,
                    "body": body.decode("utf-8", errors="replace"),
                },
            )
            await response.aclose()
            raise UpstreamAPIError(response.status_code, gemini_error_to_openai_error(response.status_code, _safe_json(body)))

        async def iterator() -> AsyncIterator[str]:
            response_id = f"resp_{uuid.uuid4().hex}"
            created_at = int(time.time())
            accumulated_text = ""
            latest_tool_calls: list[dict[str, Any]] = []
            usage_metadata: dict[str, Any] | None = None
            finish_reason: str | None = None
            created_sent = False

            async def _emit(payload_obj: dict[str, Any], *, event_name: str | None = None) -> AsyncIterator[str]:
                if event_name:
                    yield f"event: {event_name}"
                yield f"data: {json.dumps(payload_obj, ensure_ascii=False)}"
                yield ""

            try:
                async for chunk_payload in _iter_sse_json_payloads(response.aiter_lines()):
                    if not isinstance(chunk_payload, dict):
                        continue

                    if not created_sent:
                        response_id = chunk_payload.get("responseId") or response_id
                        created_at = _to_epoch(chunk_payload.get("createTime"), fallback=created_at)
                        async for line in _emit(
                            {
                                "type": "response.created",
                                "response": {
                                    "id": response_id,
                                    "object": "response",
                                    "created_at": created_at,
                                    "model": model,
                                },
                            },
                            event_name="response.created",
                        ):
                            yield line
                        created_sent = True

                    usage_value = chunk_payload.get("usageMetadata")
                    if isinstance(usage_value, dict):
                        usage_metadata = usage_value

                    candidate = first_candidate(chunk_payload)
                    finish_value = candidate.get("finishReason")
                    if isinstance(finish_value, str):
                        finish_reason = finish_value

                    current_text = extract_text_from_candidate(candidate)
                    delta_text = extract_stream_delta_text(current_text, accumulated_text)
                    if delta_text:
                        accumulated_text += delta_text
                        async for line in _emit(
                            {
                                "type": "response.output_text.delta",
                                "delta": delta_text,
                                "output_index": 0,
                            },
                            event_name="response.output_text.delta",
                        ):
                            yield line

                    current_tool_calls = extract_tool_calls_from_candidate(candidate)
                    if current_tool_calls:
                        latest_tool_calls = current_tool_calls
            except Exception as exc:
                error_payload = {
                    "type": "response.failed",
                    "response": {
                        "id": response_id,
                        "status": "failed",
                        "error": {
                            "message": f"Gemini streaming parse failed: {exc}",
                            "type": "server_error",
                            "param": None,
                            "code": "gemini_stream_parse_error",
                        },
                    },
                }
                async for line in _emit(error_payload, event_name="response.failed"):
                    yield line
                yield "data: [DONE]"
                yield ""
                return
            finally:
                await response.aclose()

            completed_response = build_openai_response_from_stream_state(
                model=model,
                response_id=response_id,
                created_at=created_at,
                full_text=accumulated_text,
                tool_calls=latest_tool_calls,
                usage_metadata=usage_metadata,
                finish_reason=finish_reason,
            )
            async for line in _emit(
                {
                    "type": "response.completed",
                    "response": completed_response,
                },
                event_name="response.completed",
            ):
                yield line
            yield "data: [DONE]"
            yield ""

        return iterator()

    async def close(self) -> None:
        await self._client.aclose()

    async def _post_with_retry(self, *, path: str, payload: dict[str, Any]) -> httpx.Response:
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                response = await self._client.post(path, json=payload)
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt >= attempts:
                    raise
                await asyncio.sleep(_retry_delay_seconds(attempt))
                continue

            if response.status_code in {429, 500, 502, 503, 504} and attempt < attempts:
                await asyncio.sleep(_retry_delay_seconds(attempt, response=response))
                continue
            return response
        return await self._client.post(path, json=payload)

    async def _send_with_retry(self, request: httpx.Request) -> httpx.Response:
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                response = await self._client.send(request, stream=True)
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt >= attempts:
                    raise
                await asyncio.sleep(_retry_delay_seconds(attempt))
                request = self._client.build_request(
                    request.method,
                    str(request.url),
                    content=request.content,
                    headers=request.headers,
                )
                continue

            if response.status_code in {429, 500, 502, 503, 504} and attempt < attempts:
                await response.aclose()
                await asyncio.sleep(_retry_delay_seconds(attempt, response=response))
                request = self._client.build_request(
                    request.method,
                    str(request.url),
                    content=request.content,
                    headers=request.headers,
                )
                continue
            return response
        return await self._client.send(request, stream=True)


class RoutingResponsesGateway(BaseResponsesGateway):
    def __init__(
        self,
        *,
        settings: Settings,
        openai_gateway: OpenAIResponsesGateway,
        gemini_gateway: GeminiResponsesGateway,
        raw_logger: RawIOLogger | None = None,
    ):
        self._settings = settings
        self._openai_gateway = openai_gateway
        self._gemini_gateway = gemini_gateway
        self._raw_logger = raw_logger or RawIOLogger.from_settings(settings)

    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        gateway = self._pick_gateway(payload)
        return await gateway.create_response(payload)

    async def stream_response(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        gateway = self._pick_gateway(payload)
        return await gateway.stream_response(payload)

    async def close(self) -> None:
        await self._openai_gateway.close()
        await self._gemini_gateway.close()

    def _pick_gateway(self, payload: dict[str, Any]) -> BaseResponsesGateway:
        model = _extract_model(payload)
        provider = "openai" if self._settings.is_openai_model(model) else "gemini"
        self._raw_logger.log(
            "proxy.route.provider",
            {"model": model, "provider": provider},
        )
        if provider == "gemini":
            return self._gemini_gateway
        return self._openai_gateway


def _extract_model(payload: dict[str, Any]) -> str:
    model = payload.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    raise UpstreamAPIError(
        400,
        {
            "error": {
                "message": "model is required.",
                "type": "invalid_request_error",
                "param": "model",
                "code": "missing_required_parameter",
            }
        },
    )


async def _iter_sse_json_payloads(lines: AsyncIterator[str]) -> AsyncIterator[dict[str, Any] | str]:
    data_lines: list[str] = []
    async for raw_line in lines:
        line = raw_line.rstrip("\r")
        if not line:
            payload = _flush_sse_data_lines(data_lines)
            if payload is not None:
                yield payload
            data_lines = []
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue
        if line.startswith(":"):
            continue

    payload = _flush_sse_data_lines(data_lines)
    if payload is not None:
        yield payload


def _flush_sse_data_lines(data_lines: list[str]) -> dict[str, Any] | str | None:
    if not data_lines:
        return None
    merged = "\n".join(data_lines)
    if merged == "[DONE]":
        return "[DONE]"
    try:
        return json.loads(merged)
    except json.JSONDecodeError:
        return {"raw": merged}


def _retry_delay_seconds(attempt: int, response: httpx.Response | None = None) -> float:
    retry_after = _parse_retry_after(response.headers.get("retry-after")) if response else None
    if retry_after is not None:
        return max(0.0, min(retry_after, 5.0))
    base = 0.25 * (2 ** (attempt - 1))
    return min(base, 2.0)


def _parse_retry_after(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    stripped = raw_value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        pass
    try:
        return max((parsedate_to_datetime(stripped) - datetime.now(parsedate_to_datetime(stripped).tzinfo)).total_seconds(), 0.0)
    except Exception:
        return None


def _to_epoch(raw_time: Any, *, fallback: int) -> int:
    if isinstance(raw_time, str):
        try:
            return int(datetime.fromisoformat(raw_time.replace("Z", "+00:00")).timestamp())
        except ValueError:
            return fallback
    return fallback


def _normalize_max_output_tokens_for_model(payload: dict[str, Any], model: str) -> dict[str, Any]:
    max_output_limit = resolve_model_max_output_tokens(model)
    if max_output_limit is None:
        return payload

    normalized = dict(payload)
    normalized["max_output_tokens"] = max_output_limit
    return normalized


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
