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
from app.services.antigravity_adapter import (
    AntigravityAdapterError,
    antigravity_error_to_openai_error,
    antigravity_message_to_openai_response,
    build_antigravity_request_from_responses,
)
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

    async def create_native_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise UpstreamAPIError(
            400,
            {
                "error": {
                    "message": "Native messages passthrough is not supported for this upstream mode.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "native_messages_unsupported",
                }
            },
        )

    async def stream_native_message(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        raise UpstreamAPIError(
            400,
            {
                "error": {
                    "message": "Native messages passthrough is not supported for this upstream mode.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "native_messages_unsupported",
                }
            },
        )


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
        if self._settings.upstream_streaming_enabled:
            stream_payload = dict(payload)
            stream_payload["stream"] = True
            stream_iter = await self.stream_response(stream_payload)
            return await _collect_completed_response_from_sse(stream_iter)
        self._raw_logger.log(
            "upstream.request",
            {"provider": "openai", "path": "/responses", "stream": False, "payload": payload},
        )
        response = await _post_json_with_retry(
            client=self._client,
            path="/responses",
            payload=payload,
            provider="openai",
            stream=False,
            raw_logger=self._raw_logger,
        )
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
        response = await _send_request_with_retry(
            client=self._client,
            request=request,
            provider="openai",
            path="/responses",
            stream=True,
            raw_logger=self._raw_logger,
        )
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
                try:
                    async for line in response.aiter_lines():
                        self._raw_logger.log(
                            "upstream.response.stream_line",
                            {"provider": "openai", "path": "/responses", "line": line},
                        )
                        yield line
                except httpx.HTTPError as exc:
                    error = _wrap_transport_exception(
                        exc,
                        provider="openai",
                        path="/responses",
                        stream=True,
                        raw_logger=self._raw_logger,
                    )
                    for line in _build_response_failed_sse_lines(error):
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
        self._request_interval_seconds = float(settings.gemini_min_request_interval_seconds)
        self._request_lock = asyncio.Lock()
        self._last_request_started_at = 0.0
        self._client = httpx.AsyncClient(
            base_url=settings.upstream_gemini_base_url,
            timeout=settings.upstream_timeout_seconds,
            headers={
                "x-goog-api-key": settings.upstream_gemini_api_key,
                "Content-Type": "application/json",
            },
        )

    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
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

        requested_model = _extract_model(payload)
        try:
            return await self._create_response_for_model(
                payload=payload,
                target_model=requested_model,
                response_model=requested_model,
            )
        except UpstreamAPIError as primary_exc:
            fallback_model = self._resolve_fallback_model(requested_model, primary_exc)
            if not fallback_model:
                raise
            self._raw_logger.log(
                "upstream.fallback",
                {
                    "provider": "gemini",
                    "from_model": requested_model,
                    "to_model": fallback_model,
                    "status_code": primary_exc.status_code,
                    "error": primary_exc.payload,
                    "stream": False,
                },
            )
            return await self._create_response_for_model(
                payload=payload,
                target_model=fallback_model,
                response_model=requested_model,
            )

    async def _create_response_for_model(
        self,
        *,
        payload: dict[str, Any],
        target_model: str,
        response_model: str,
    ) -> dict[str, Any]:
        normalized_payload = _normalize_max_output_tokens_for_model(payload, target_model)
        path = f"/models/{target_model}:generateContent"
        parsed = await self._perform_non_stream_request(
            normalized_payload=normalized_payload,
            path=path,
            target_model=target_model,
            response_model=response_model,
        )

        return gemini_response_to_openai_response(parsed, model=response_model)

    async def stream_response(self, payload: dict[str, Any]) -> AsyncIterator[str]:
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

        requested_model = _extract_model(payload)
        try:
            return await self._stream_response_for_model(
                payload=payload,
                target_model=requested_model,
                response_model=requested_model,
            )
        except UpstreamAPIError as primary_exc:
            fallback_model = self._resolve_fallback_model(requested_model, primary_exc)
            if not fallback_model:
                raise
            self._raw_logger.log(
                "upstream.fallback",
                {
                    "provider": "gemini",
                    "from_model": requested_model,
                    "to_model": fallback_model,
                    "status_code": primary_exc.status_code,
                    "error": primary_exc.payload,
                    "stream": True,
                },
            )
            return await self._stream_response_for_model(
                payload=payload,
                target_model=fallback_model,
                response_model=requested_model,
            )

    async def _stream_response_for_model(
        self,
        *,
        payload: dict[str, Any],
        target_model: str,
        response_model: str,
    ) -> AsyncIterator[str]:
        normalized_payload = _normalize_max_output_tokens_for_model(payload, target_model)
        path = f"/models/{target_model}:streamGenerateContent?alt=sse"
        response = await self._perform_stream_request(
            normalized_payload=normalized_payload,
            path=path,
            target_model=target_model,
            response_model=response_model,
        )

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
                                    "model": response_model,
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
                model=response_model,
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

    def _resolve_fallback_model(self, requested_model: str, error: UpstreamAPIError) -> str | None:
        fallback_model = self._settings.upstream_fallback_model
        if not fallback_model:
            return None
        if fallback_model.strip().lower() == requested_model.strip().lower():
            return None
        if error.status_code in {429, 500, 502, 503, 504}:
            return fallback_model

        payload = error.payload
        if isinstance(payload, dict):
            error_obj = payload.get("error")
            if isinstance(error_obj, dict):
                message = error_obj.get("message")
                if isinstance(message, str) and "No available Gemini accounts" in message:
                    return fallback_model
        return None

    async def close(self) -> None:
        await self._client.aclose()

    async def _wait_for_rate_slot(self) -> None:
        if self._request_interval_seconds <= 0:
            return
        async with self._request_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_started_at
            wait_seconds = self._request_interval_seconds - elapsed
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
                now = time.monotonic()
            self._last_request_started_at = now

    async def _post_with_retry(self, *, path: str, payload: dict[str, Any]) -> httpx.Response:
        return await _post_json_with_retry(
            client=self._client,
            path=path,
            payload=payload,
            provider="gemini",
            stream=False,
            raw_logger=self._raw_logger,
        )

    async def _send_with_retry(self, request: httpx.Request) -> httpx.Response:
        return await _send_request_with_retry(
            client=self._client,
            request=request,
            provider="gemini",
            path=str(request.url),
            stream=True,
            raw_logger=self._raw_logger,
        )

    async def _perform_non_stream_request(
        self,
        *,
        normalized_payload: dict[str, Any],
        path: str,
        target_model: str,
        response_model: str,
    ) -> dict[str, Any]:
        tried_without_max = False
        payload_variant = dict(normalized_payload)

        while True:
            try:
                gemini_payload = build_gemini_request_from_responses(payload_variant)
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

            self._raw_logger.log(
                "upstream.request",
                {
                    "provider": "gemini",
                    "path": path,
                    "stream": False,
                    "target_model": target_model,
                    "response_model": response_model,
                    "payload": gemini_payload,
                },
            )
            await self._wait_for_rate_slot()
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
                if (
                    not tried_without_max
                    and "max_output_tokens" in payload_variant
                    and _is_invalid_argument_error(response.status_code, raw_error)
                ):
                    tried_without_max = True
                    payload_variant = dict(payload_variant)
                    payload_variant.pop("max_output_tokens", None)
                    self._raw_logger.log(
                        "upstream.request.retry_without_max_output_tokens",
                        {
                            "provider": "gemini",
                            "path": path,
                            "target_model": target_model,
                            "response_model": response_model,
                            "status_code": response.status_code,
                        },
                    )
                    continue
                raise UpstreamAPIError(
                    response.status_code,
                    gemini_error_to_openai_error(response.status_code, raw_error),
                )

            try:
                return response.json()
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

    async def _perform_stream_request(
        self,
        *,
        normalized_payload: dict[str, Any],
        path: str,
        target_model: str,
        response_model: str,
    ) -> httpx.Response:
        tried_without_max = False
        payload_variant = dict(normalized_payload)

        while True:
            try:
                gemini_payload = build_gemini_request_from_responses(payload_variant)
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

            self._raw_logger.log(
                "upstream.request",
                {
                    "provider": "gemini",
                    "path": path,
                    "stream": True,
                    "target_model": target_model,
                    "response_model": response_model,
                    "payload": gemini_payload,
                },
            )
            request = self._client.build_request(
                "POST",
                path,
                json=gemini_payload,
                headers={"Accept": "text/event-stream"},
            )
            await self._wait_for_rate_slot()
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
            if response.status_code < 400:
                return response

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
            raw_error = _safe_json(body)
            if (
                not tried_without_max
                and "max_output_tokens" in payload_variant
                and _is_invalid_argument_error(response.status_code, raw_error)
            ):
                tried_without_max = True
                payload_variant = dict(payload_variant)
                payload_variant.pop("max_output_tokens", None)
                self._raw_logger.log(
                    "upstream.request.retry_without_max_output_tokens",
                    {
                        "provider": "gemini",
                        "path": path,
                        "target_model": target_model,
                        "response_model": response_model,
                        "status_code": response.status_code,
                        "stream": True,
                    },
                )
                continue

            raise UpstreamAPIError(
                response.status_code,
                gemini_error_to_openai_error(response.status_code, raw_error),
            )


class AntigravityResponsesGateway(BaseResponsesGateway):
    def __init__(self, settings: Settings, raw_logger: RawIOLogger | None = None):
        self._settings = settings
        self._raw_logger = raw_logger or RawIOLogger.from_settings(settings)
        self._request_interval_seconds = float(settings.upstream_min_request_interval_seconds)
        self._request_lock = asyncio.Lock()
        self._last_request_started_at = 0.0
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "x-api-key": settings.upstream_api_key,
            "anthropic-version": settings.upstream_messages_api_version,
        }
        self._client = httpx.AsyncClient(
            base_url=settings.upstream_base_url,
            timeout=settings.upstream_timeout_seconds,
            headers=headers,
        )

    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._settings.upstream_api_key:
            raise UpstreamAPIError(
                500,
                {
                    "error": {
                        "message": "UPSTREAM_API_KEY is required for messages mode.",
                        "type": "server_error",
                        "param": None,
                        "code": "upstream_api_key_missing",
                    }
                },
            )
        if self._settings.upstream_streaming_enabled:
            stream_iter = await self.stream_response(payload)
            return await _collect_completed_response_from_sse(stream_iter)

        requested_model = _extract_model(payload)
        try:
            return await self._create_response_for_model(
                payload=payload,
                target_model=requested_model,
                response_model=requested_model,
            )
        except UpstreamAPIError as primary_exc:
            fallback_model = self._resolve_fallback_model(requested_model, primary_exc)
            if not fallback_model:
                raise
            self._raw_logger.log(
                "upstream.fallback",
                {
                    "provider": "antigravity",
                    "from_model": requested_model,
                    "to_model": fallback_model,
                    "status_code": primary_exc.status_code,
                    "error": primary_exc.payload,
                    "stream": False,
                },
            )
            return await self._create_response_for_model(
                payload=payload,
                target_model=fallback_model,
                response_model=requested_model,
            )

    async def create_native_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        requested_model = _extract_native_message_model(payload, self._settings.default_upstream_model)
        try:
            return await self._create_native_message_for_model(
                payload=payload,
                target_model=requested_model,
            )
        except UpstreamAPIError as primary_exc:
            fallback_model = self._resolve_fallback_model(requested_model, primary_exc)
            if not fallback_model:
                raise
            self._raw_logger.log(
                "upstream.fallback",
                {
                    "provider": "antigravity",
                    "from_model": requested_model,
                    "to_model": fallback_model,
                    "status_code": primary_exc.status_code,
                    "error": primary_exc.payload,
                    "stream": False,
                    "native_messages": True,
                },
            )
            return await self._create_native_message_for_model(
                payload=payload,
                target_model=fallback_model,
            )

    async def _create_response_for_model(
        self,
        *,
        payload: dict[str, Any],
        target_model: str,
        response_model: str,
    ) -> dict[str, Any]:
        normalized_payload = _normalize_max_output_tokens_for_model(payload, target_model)
        payload_variant = dict(normalized_payload)
        payload_variant["model"] = target_model
        path = "/messages"
        raw_response = await self._perform_non_stream_request(
            payload_variant=payload_variant,
            path=path,
            target_model=target_model,
            response_model=response_model,
        )
        return antigravity_message_to_openai_response(raw_response, model=response_model)

    async def _create_native_message_for_model(
        self,
        *,
        payload: dict[str, Any],
        target_model: str,
    ) -> dict[str, Any]:
        path = _extract_native_message_path(payload)
        forward_headers = _extract_native_forward_headers(payload)
        request_payload = _prepare_native_message_payload(
            payload=payload,
            target_model=target_model,
            settings=self._settings,
        )
        return await self._perform_non_stream_antigravity_request(
            antigravity_payload=request_payload,
            path=path,
            target_model=target_model,
            response_model=target_model,
            native_messages=True,
            forwarded_headers=forward_headers,
        )

    async def stream_response(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        if not self._settings.upstream_api_key:
            raise UpstreamAPIError(
                500,
                {
                    "error": {
                        "message": "UPSTREAM_API_KEY is required for messages mode.",
                        "type": "server_error",
                        "param": None,
                        "code": "upstream_api_key_missing",
                    }
                },
            )

        requested_model = _extract_model(payload)
        try:
            return await self._stream_response_for_model(
                payload=payload,
                target_model=requested_model,
                response_model=requested_model,
            )
        except UpstreamAPIError as primary_exc:
            fallback_model = self._resolve_fallback_model(requested_model, primary_exc)
            if not fallback_model:
                raise
            self._raw_logger.log(
                "upstream.fallback",
                {
                    "provider": "antigravity",
                    "from_model": requested_model,
                    "to_model": fallback_model,
                    "status_code": primary_exc.status_code,
                    "error": primary_exc.payload,
                    "stream": True,
                },
            )
            return await self._stream_response_for_model(
                payload=payload,
                target_model=fallback_model,
                response_model=requested_model,
            )

    async def stream_native_message(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        requested_model = _extract_native_message_model(payload, self._settings.default_upstream_model)
        try:
            return await self._stream_native_message_for_model(
                payload=payload,
                target_model=requested_model,
            )
        except UpstreamAPIError as primary_exc:
            fallback_model = self._resolve_fallback_model(requested_model, primary_exc)
            if not fallback_model:
                raise
            self._raw_logger.log(
                "upstream.fallback",
                {
                    "provider": "antigravity",
                    "from_model": requested_model,
                    "to_model": fallback_model,
                    "status_code": primary_exc.status_code,
                    "error": primary_exc.payload,
                    "stream": True,
                    "native_messages": True,
                },
            )
            return await self._stream_native_message_for_model(
                payload=payload,
                target_model=fallback_model,
            )

    async def _stream_response_for_model(
        self,
        *,
        payload: dict[str, Any],
        target_model: str,
        response_model: str,
    ) -> AsyncIterator[str]:
        normalized_payload = _normalize_max_output_tokens_for_model(payload, target_model)
        payload_variant = dict(normalized_payload)
        payload_variant["model"] = target_model
        payload_variant["stream"] = True
        path = "/messages"
        response = await self._perform_stream_request(
            payload_variant=payload_variant,
            path=path,
            target_model=target_model,
            response_model=response_model,
        )

        async def iterator() -> AsyncIterator[str]:
            response_id = f"resp_{uuid.uuid4().hex}"
            created_at = int(time.time())
            accumulated_text = ""
            usage_state: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
            finish_reason: str | None = None
            created_sent = False
            message_item_id = f"msg_{uuid.uuid4().hex}"
            message_output_index: int | None = None
            next_output_index = 0
            tool_state_by_index: dict[int, dict[str, Any]] = {}
            emitted_tool_ids: set[str] = set()
            completed_tool_calls: list[dict[str, Any]] = []

            async def _emit(payload_obj: dict[str, Any], *, event_name: str | None = None) -> AsyncIterator[str]:
                if event_name:
                    yield f"event: {event_name}"
                yield f"data: {json.dumps(payload_obj, ensure_ascii=False)}"
                yield ""

            def _usage_from_raw(raw_usage: Any) -> dict[str, int]:
                if not isinstance(raw_usage, dict):
                    return {"input_tokens": 0, "output_tokens": 0}
                return {
                    "input_tokens": _safe_int(raw_usage.get("input_tokens")),
                    "output_tokens": _safe_int(raw_usage.get("output_tokens")),
                }

            def _build_created_response() -> dict[str, Any]:
                return {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "status": "in_progress",
                    "model": response_model,
                    "output": [],
                    "output_text": "",
                    "usage": {
                        "input_tokens": usage_state["input_tokens"],
                        "output_tokens": usage_state["output_tokens"],
                        "total_tokens": usage_state["input_tokens"] + usage_state["output_tokens"],
                    },
                }

            try:
                async for event_payload in _iter_sse_json_payloads(response.aiter_lines()):
                    if not isinstance(event_payload, dict):
                        continue

                    event_type = event_payload.get("type")
                    if not isinstance(event_type, str):
                        continue

                    if event_type == "message_start":
                        message_obj = event_payload.get("message")
                        if isinstance(message_obj, dict):
                            upstream_id = message_obj.get("id")
                            if isinstance(upstream_id, str) and upstream_id.strip():
                                response_id = upstream_id
                            usage_obj = _usage_from_raw(message_obj.get("usage"))
                            usage_state["input_tokens"] = usage_obj["input_tokens"]
                            usage_state["output_tokens"] = usage_obj["output_tokens"]

                        if not created_sent:
                            async for line in _emit(
                                {
                                    "type": "response.created",
                                    "response": _build_created_response(),
                                },
                                event_name="response.created",
                            ):
                                yield line
                            created_sent = True
                        continue

                    if not created_sent:
                        async for line in _emit(
                            {
                                "type": "response.created",
                                "response": _build_created_response(),
                            },
                            event_name="response.created",
                        ):
                            yield line
                        created_sent = True

                    if event_type == "content_block_start":
                        index = _safe_int(event_payload.get("index"))
                        block = event_payload.get("content_block")
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") != "tool_use":
                            continue
                        call_id = block.get("id")
                        if not isinstance(call_id, str) or not call_id.strip():
                            call_id = f"call_{uuid.uuid4().hex}"
                        function_name = block.get("name")
                        if not isinstance(function_name, str) or not function_name.strip():
                            function_name = "tool"
                        raw_input = block.get("input")
                        initial_input = raw_input if isinstance(raw_input, dict) else None
                        item_id = f"item_{call_id}"
                        output_index = next_output_index
                        next_output_index += 1
                        state = {
                            "item_id": item_id,
                            "call_id": call_id,
                            "name": function_name,
                            "arguments": "",
                            "initial_input": initial_input,
                            "output_index": output_index,
                        }
                        tool_state_by_index[index] = state
                        if call_id not in emitted_tool_ids:
                            emitted_tool_ids.add(call_id)
                            async for line in _emit(
                                {
                                    "type": "response.output_item.added",
                                    "item": {
                                        "type": "function_call",
                                        "id": item_id,
                                        "call_id": call_id,
                                        "name": function_name,
                                        "arguments": "",
                                    },
                                    "output_index": output_index,
                                },
                                event_name="response.output_item.added",
                            ):
                                yield line
                        continue

                    if event_type == "content_block_delta":
                        index = _safe_int(event_payload.get("index"))
                        delta = event_payload.get("delta")
                        if not isinstance(delta, dict):
                            continue

                        delta_type = delta.get("type")
                        if delta_type == "text_delta":
                            text_delta = delta.get("text")
                            if isinstance(text_delta, str) and text_delta:
                                if message_output_index is None:
                                    message_output_index = next_output_index
                                    next_output_index += 1
                                    async for line in _emit(
                                        {
                                            "type": "response.output_item.added",
                                            "item": {
                                                "id": message_item_id,
                                                "type": "message",
                                                "role": "assistant",
                                                "status": "in_progress",
                                                "content": [
                                                    {
                                                        "type": "output_text",
                                                        "text": "",
                                                        "annotations": [],
                                                    }
                                                ],
                                            },
                                            "output_index": message_output_index,
                                        },
                                        event_name="response.output_item.added",
                                    ):
                                        yield line
                                accumulated_text += text_delta
                                async for line in _emit(
                                    {
                                        "type": "response.output_text.delta",
                                        "delta": text_delta,
                                        "item_id": message_item_id,
                                        "output_index": message_output_index,
                                        "content_index": 0,
                                    },
                                    event_name="response.output_text.delta",
                                ):
                                    yield line
                            continue

                        if delta_type == "input_json_delta":
                            state = tool_state_by_index.get(index)
                            if not state:
                                continue
                            partial_json = delta.get("partial_json")
                            if not isinstance(partial_json, str) or not partial_json:
                                continue
                            current_arguments = state.get("arguments")
                            if not isinstance(current_arguments, str):
                                current_arguments = ""
                            state["arguments"] = f"{current_arguments}{partial_json}"
                            async for line in _emit(
                                {
                                    "type": "response.function_call_arguments.delta",
                                    "item_id": state["item_id"],
                                    "delta": partial_json,
                                },
                                event_name="response.function_call_arguments.delta",
                            ):
                                yield line
                            continue

                    if event_type == "content_block_stop":
                        index = _safe_int(event_payload.get("index"))
                        state = tool_state_by_index.get(index)
                        if not state:
                            continue
                        arguments_text = state.get("arguments")
                        if not isinstance(arguments_text, str):
                            arguments_text = ""
                        if not arguments_text:
                            initial_input = state.get("initial_input")
                            if isinstance(initial_input, dict):
                                arguments_text = json.dumps(
                                    initial_input,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                        completed_tool_calls.append(
                            {
                                "output_index": int(state.get("output_index", 0)),
                                "type": "function_call",
                                "id": state["item_id"],
                                "call_id": state["call_id"],
                                "name": state["name"],
                                "arguments": arguments_text,
                            }
                        )
                        async for line in _emit(
                            {
                                "type": "response.function_call_arguments.done",
                                "item_id": state["item_id"],
                                "arguments": arguments_text,
                            },
                            event_name="response.function_call_arguments.done",
                        ):
                            yield line
                        async for line in _emit(
                            {
                                "type": "response.output_item.done",
                                    "item": {
                                        "type": "function_call",
                                        "id": state["item_id"],
                                        "call_id": state["call_id"],
                                        "name": state["name"],
                                        "arguments": arguments_text,
                                    },
                                    "output_index": int(state.get("output_index", 0)),
                                },
                                event_name="response.output_item.done",
                            ):
                                yield line
                        continue

                    if event_type == "message_delta":
                        usage_obj = _usage_from_raw(event_payload.get("usage"))
                        usage_state["input_tokens"] = max(usage_state["input_tokens"], usage_obj["input_tokens"])
                        usage_state["output_tokens"] = max(usage_state["output_tokens"], usage_obj["output_tokens"])
                        delta_obj = event_payload.get("delta")
                        if isinstance(delta_obj, dict):
                            stop_value = delta_obj.get("stop_reason")
                            if isinstance(stop_value, str):
                                finish_reason = stop_value
                        continue

                    if event_type == "error":
                        error_payload = antigravity_error_to_openai_error(502, event_payload)
                        async for line in _emit(
                            {
                                "type": "response.failed",
                                "response": {
                                    "id": response_id,
                                    "status": "failed",
                                    "error": error_payload.get("error"),
                                },
                            },
                            event_name="response.failed",
                        ):
                            yield line
                        yield "data: [DONE]"
                        yield ""
                        return
            except Exception as exc:
                error_payload = {
                    "type": "response.failed",
                    "response": {
                        "id": response_id,
                        "status": "failed",
                        "error": {
                            "message": f"Antigravity streaming parse failed: {exc}",
                            "type": "server_error",
                            "param": None,
                            "code": "antigravity_stream_parse_error",
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

            usage = {
                "input_tokens": usage_state["input_tokens"],
                "output_tokens": usage_state["output_tokens"],
                "total_tokens": usage_state["input_tokens"] + usage_state["output_tokens"],
            }
            output_entries: list[tuple[int, dict[str, Any]]] = []
            if message_output_index is not None:
                message_item = (
                    {
                        "id": message_item_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": accumulated_text,
                                "annotations": [],
                            }
                        ],
                    }
                )
                output_entries.append((message_output_index, message_item))
                async for line in _emit(
                    {
                        "type": "response.output_item.done",
                        "item": message_item,
                        "output_index": message_output_index,
                    },
                    event_name="response.output_item.done",
                ):
                    yield line
            for tool_item in completed_tool_calls:
                output_index = int(tool_item.get("output_index", 0))
                tool_output_item = dict(tool_item)
                tool_output_item.pop("output_index", None)
                output_entries.append((output_index, tool_output_item))

            output_items = [item for _, item in sorted(output_entries, key=lambda pair: pair[0])]

            completed_response: dict[str, Any] = {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "status": "completed",
                "model": response_model,
                "output": output_items,
                "output_text": accumulated_text,
                "usage": usage,
            }
            if finish_reason == "max_tokens":
                completed_response["incomplete_details"] = {"reason": "max_output_tokens"}

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

    async def _stream_native_message_for_model(
        self,
        *,
        payload: dict[str, Any],
        target_model: str,
    ) -> AsyncIterator[str]:
        path = _extract_native_message_path(payload)
        forward_headers = _extract_native_forward_headers(payload)
        request_payload = _prepare_native_message_payload(
            payload=payload,
            target_model=target_model,
            settings=self._settings,
        )
        request_payload["stream"] = True
        response = await self._perform_stream_antigravity_request(
            antigravity_payload=request_payload,
            path=path,
            target_model=target_model,
            response_model=target_model,
            native_messages=True,
            forwarded_headers=forward_headers,
        )

        async def iterator() -> AsyncIterator[str]:
            try:
                async for line in response.aiter_lines():
                    self._raw_logger.log(
                        "upstream.response.stream_line",
                        {
                            "provider": "antigravity",
                            "path": path,
                            "native_messages": True,
                            "line": line,
                        },
                    )
                    yield line
            finally:
                await response.aclose()

        return iterator()

    def _resolve_fallback_model(self, requested_model: str, error: UpstreamAPIError) -> str | None:
        fallback_model = self._settings.upstream_fallback_model
        if not fallback_model:
            return None
        if fallback_model.strip().lower() == requested_model.strip().lower():
            return None
        if error.status_code in {429, 500, 502, 503, 504}:
            return fallback_model
        payload = error.payload
        if isinstance(payload, dict):
            error_obj = payload.get("error")
            if isinstance(error_obj, dict):
                message = error_obj.get("message")
                if isinstance(message, str) and "no available" in message.lower():
                    return fallback_model
        return None

    async def close(self) -> None:
        await self._client.aclose()

    async def _wait_for_rate_slot(self) -> None:
        if self._request_interval_seconds <= 0:
            return
        async with self._request_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_started_at
            wait_seconds = self._request_interval_seconds - elapsed
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
                now = time.monotonic()
            self._last_request_started_at = now

    async def _post_with_retry(
        self,
        *,
        path: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return await _post_json_with_retry(
            client=self._client,
            path=path,
            payload=payload,
            provider="antigravity",
            stream=False,
            raw_logger=self._raw_logger,
            headers=headers,
        )

    async def _send_with_retry(self, request: httpx.Request) -> httpx.Response:
        return await _send_request_with_retry(
            client=self._client,
            request=request,
            provider="antigravity",
            path=str(request.url),
            stream=True,
            raw_logger=self._raw_logger,
        )
        return await self._client.send(request, stream=True)

    async def _perform_non_stream_request(
        self,
        *,
        payload_variant: dict[str, Any],
        path: str,
        target_model: str,
        response_model: str,
    ) -> dict[str, Any]:
        try:
            antigravity_payload = build_antigravity_request_from_responses(payload_variant)
        except AntigravityAdapterError as exc:
            raise UpstreamAPIError(
                400,
                {
                    "error": {
                        "message": str(exc),
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_antigravity_request",
                    }
                },
            ) from exc
        return await self._perform_non_stream_antigravity_request(
            antigravity_payload=antigravity_payload,
            path=path,
            target_model=target_model,
            response_model=response_model,
            native_messages=False,
            forwarded_headers=None,
        )

    async def _perform_non_stream_antigravity_request(
        self,
        *,
        antigravity_payload: dict[str, Any],
        path: str,
        target_model: str,
        response_model: str,
        native_messages: bool,
        forwarded_headers: dict[str, str] | None,
    ) -> dict[str, Any]:
        self._raw_logger.log(
            "upstream.request",
            {
                "provider": "antigravity",
                "path": path,
                "stream": False,
                "target_model": target_model,
                "response_model": response_model,
                "native_messages": native_messages,
                "payload": antigravity_payload,
            },
        )
        await self._wait_for_rate_slot()
        extra_headers = self._extra_headers_for_payload(
            payload=antigravity_payload,
            forwarded_headers=forwarded_headers,
        )
        response = await self._post_with_retry(
            path=path,
            payload=antigravity_payload,
            headers=extra_headers or None,
        )
        self._raw_logger.log(
            "upstream.response",
            {
                "provider": "antigravity",
                "path": path,
                "stream": False,
                "native_messages": native_messages,
                "status_code": response.status_code,
                "body": response.text,
            },
        )
        if response.status_code >= 400:
            raw_error = _safe_json(response)
            raise UpstreamAPIError(
                response.status_code,
                raw_error if native_messages else antigravity_error_to_openai_error(response.status_code, raw_error),
            )

        try:
            parsed = response.json()
        except ValueError:
            raise UpstreamAPIError(
                502,
                {
                    "error": {
                        "message": "Antigravity upstream returned non-JSON success response.",
                        "type": "upstream_invalid_response",
                        "param": None,
                        "code": "invalid_upstream_response",
                    }
                },
            ) from None

        if not isinstance(parsed, dict):
            raise UpstreamAPIError(
                502,
                {
                    "error": {
                        "message": "Antigravity upstream returned non-object JSON response.",
                        "type": "upstream_invalid_response",
                        "param": None,
                        "code": "invalid_upstream_response",
                    }
                },
            )
        return parsed

    async def _perform_stream_request(
        self,
        *,
        payload_variant: dict[str, Any],
        path: str,
        target_model: str,
        response_model: str,
    ) -> httpx.Response:
        try:
            antigravity_payload = build_antigravity_request_from_responses(payload_variant)
        except AntigravityAdapterError as exc:
            raise UpstreamAPIError(
                400,
                {
                    "error": {
                        "message": str(exc),
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_antigravity_request",
                    }
                },
            ) from exc
        return await self._perform_stream_antigravity_request(
            antigravity_payload=antigravity_payload,
            path=path,
            target_model=target_model,
            response_model=response_model,
            native_messages=False,
            forwarded_headers=None,
        )

    async def _perform_stream_antigravity_request(
        self,
        *,
        antigravity_payload: dict[str, Any],
        path: str,
        target_model: str,
        response_model: str,
        native_messages: bool,
        forwarded_headers: dict[str, str] | None,
    ) -> httpx.Response:
        self._raw_logger.log(
            "upstream.request",
            {
                "provider": "antigravity",
                "path": path,
                "stream": True,
                "target_model": target_model,
                "response_model": response_model,
                "native_messages": native_messages,
                "payload": antigravity_payload,
            },
        )
        request = self._client.build_request(
            "POST",
            path,
            json=antigravity_payload,
            headers={
                "Accept": "text/event-stream",
                **self._extra_headers_for_payload(
                    payload=antigravity_payload,
                    forwarded_headers=forwarded_headers,
                ),
            },
        )
        await self._wait_for_rate_slot()
        response = await self._send_with_retry(request)
        self._raw_logger.log(
            "upstream.response",
            {
                "provider": "antigravity",
                "path": path,
                "stream": True,
                "native_messages": native_messages,
                "status_code": response.status_code,
            },
        )
        if response.status_code < 400:
            return response

        body = await response.aread()
        self._raw_logger.log(
            "upstream.response.error_body",
            {
                "provider": "antigravity",
                "path": path,
                "stream": True,
                "native_messages": native_messages,
                "status_code": response.status_code,
                "body": body.decode("utf-8", errors="replace"),
            },
        )
        await response.aclose()
        raw_error = _safe_json(body)
        raise UpstreamAPIError(
            response.status_code,
            raw_error if native_messages else antigravity_error_to_openai_error(response.status_code, raw_error),
        )

    def _extra_headers_for_payload(
        self,
        *,
        payload: dict[str, Any],
        forwarded_headers: dict[str, str] | None,
    ) -> dict[str, str]:
        headers = dict(forwarded_headers or {})
        speed = payload.get("speed")
        if isinstance(speed, str) and speed.strip().lower() == "fast":
            existing_beta = headers.get("anthropic-beta", "").strip()
            beta_values = [item.strip() for item in existing_beta.split(",") if item.strip()]
            if "fast-mode-2026-02-01" not in beta_values:
                beta_values.append("fast-mode-2026-02-01")
            headers["anthropic-beta"] = ",".join(beta_values)
        return headers


class RoutingResponsesGateway(BaseResponsesGateway):
    def __init__(
        self,
        *,
        settings: Settings,
        upstream_gateway: BaseResponsesGateway,
        raw_logger: RawIOLogger | None = None,
    ):
        self._settings = settings
        self._upstream_gateway = upstream_gateway
        self._raw_logger = raw_logger or RawIOLogger.from_settings(settings)

    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        attempts = self._build_attempt_payloads(payload)
        last_error: UpstreamAPIError | None = None
        for index, attempt_payload in enumerate(attempts):
            try:
                return await self._upstream_gateway.create_response(attempt_payload)
            except UpstreamAPIError as exc:
                last_error = exc
                has_next = index < len(attempts) - 1
                if not has_next or not _is_retryable_upstream_error(exc):
                    raise
                self._raw_logger.log(
                    "proxy.route.fallback",
                    {
                        "from_model": attempt_payload.get("model"),
                        "to_model": attempts[index + 1].get("model"),
                        "status_code": exc.status_code,
                        "error": exc.payload,
                        "stream": False,
                    },
                )
        if last_error is not None:
            raise last_error
        raise UpstreamAPIError(
            500,
            {
                "error": {
                    "message": "No available upstream model candidates.",
                    "type": "server_error",
                    "param": None,
                    "code": "no_model_candidates",
                }
            },
        )

    async def stream_response(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        attempts = self._build_attempt_payloads(payload)
        last_error: UpstreamAPIError | None = None
        for index, attempt_payload in enumerate(attempts):
            try:
                return await self._upstream_gateway.stream_response(attempt_payload)
            except UpstreamAPIError as exc:
                last_error = exc
                has_next = index < len(attempts) - 1
                if not has_next or not _is_retryable_upstream_error(exc):
                    raise
                self._raw_logger.log(
                    "proxy.route.fallback",
                    {
                        "from_model": attempt_payload.get("model"),
                        "to_model": attempts[index + 1].get("model"),
                        "status_code": exc.status_code,
                        "error": exc.payload,
                        "stream": True,
                    },
                )
        if last_error is not None:
            raise last_error
        raise UpstreamAPIError(
            500,
            {
                "error": {
                    "message": "No available upstream model candidates.",
                    "type": "server_error",
                    "param": None,
                    "code": "no_model_candidates",
                }
            },
        )

    async def create_native_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        attempts = self._build_native_message_attempt_payloads(payload)
        last_error: UpstreamAPIError | None = None
        for index, attempt_payload in enumerate(attempts):
            try:
                return await self._upstream_gateway.create_native_message(attempt_payload)
            except UpstreamAPIError as exc:
                last_error = exc
                has_next = index < len(attempts) - 1
                if not has_next or not _is_retryable_upstream_error(exc):
                    raise
                self._raw_logger.log(
                    "proxy.route.fallback",
                    {
                        "from_model": attempt_payload.get("model"),
                        "to_model": attempts[index + 1].get("model"),
                        "status_code": exc.status_code,
                        "error": exc.payload,
                        "stream": False,
                        "native_messages": True,
                    },
                )
        if last_error is not None:
            raise last_error
        raise UpstreamAPIError(
            500,
            {
                "error": {
                    "message": "No available upstream model candidates.",
                    "type": "server_error",
                    "param": None,
                    "code": "no_model_candidates",
                }
            },
        )

    async def stream_native_message(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        attempts = self._build_native_message_attempt_payloads(payload)
        last_error: UpstreamAPIError | None = None
        for index, attempt_payload in enumerate(attempts):
            try:
                return await self._upstream_gateway.stream_native_message(attempt_payload)
            except UpstreamAPIError as exc:
                last_error = exc
                has_next = index < len(attempts) - 1
                if not has_next or not _is_retryable_upstream_error(exc):
                    raise
                self._raw_logger.log(
                    "proxy.route.fallback",
                    {
                        "from_model": attempt_payload.get("model"),
                        "to_model": attempts[index + 1].get("model"),
                        "status_code": exc.status_code,
                        "error": exc.payload,
                        "stream": True,
                        "native_messages": True,
                    },
                )
        if last_error is not None:
            raise last_error
        raise UpstreamAPIError(
            500,
            {
                "error": {
                    "message": "No available upstream model candidates.",
                    "type": "server_error",
                    "param": None,
                    "code": "no_model_candidates",
                }
            },
        )

    async def close(self) -> None:
        await self._upstream_gateway.close()

    def _build_attempt_payloads(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        force_chain = self._settings.force_model_chain()
        base_payload = dict(payload)
        base_payload["model"] = self._settings.default_upstream_model
        base_payload.pop("speed", None)
        base_payload.pop("service_tier", None)
        if self._settings.default_upstream_speed:
            normalized_speed = self._settings.default_upstream_speed.strip().lower()
            if self._settings.upstream_mode == "messages" and normalized_speed == "fast":
                base_payload["speed"] = normalized_speed
            elif normalized_speed == "fast":
                base_payload["service_tier"] = "priority"
        if not force_chain:
            return [base_payload]

        attempts: list[dict[str, Any]] = []
        for model in force_chain:
            item = dict(base_payload)
            item["model"] = model
            attempts.append(item)
        self._raw_logger.log(
            "proxy.route.force_chain",
            {
                "enabled": True,
                "models": list(force_chain),
                "count": len(attempts),
            },
        )
        return attempts

    def _build_native_message_attempt_payloads(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        force_chain = self._settings.force_model_chain()
        models = force_chain or (self._settings.default_upstream_model,)
        attempts = [
            _prepare_native_message_payload(
                payload=payload,
                target_model=model,
                settings=self._settings,
            )
            for model in models
        ]
        if force_chain:
            self._raw_logger.log(
                "proxy.route.force_chain",
                {
                    "enabled": True,
                    "models": list(force_chain),
                    "count": len(attempts),
                    "native_messages": True,
                },
            )
        return attempts


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


def _extract_native_message_model(payload: dict[str, Any], fallback_model: str) -> str:
    model = payload.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return fallback_model


def _prepare_native_message_payload(
    *,
    payload: dict[str, Any],
    target_model: str,
    settings: Settings,
) -> dict[str, Any]:
    base_payload = dict(payload)
    base_payload.pop("__proxy_query_string", None)
    base_payload.pop("__proxy_forward_headers", None)
    base_payload["model"] = target_model
    base_payload.pop("user", None)
    base_payload.pop("service_tier", None)
    base_payload.pop("speed", None)

    default_speed = settings.default_upstream_speed
    if isinstance(default_speed, str) and default_speed.strip().lower() == "fast":
        base_payload["speed"] = "fast"

    reasoning_effort = settings.reasoning_effort_for_model(target_model)
    if reasoning_effort:
        _apply_native_message_reasoning(
            payload=base_payload,
            model=target_model,
            reasoning_effort=reasoning_effort,
        )
    else:
        base_payload.pop("reasoning", None)
        base_payload.pop("output_config", None)

    if _should_strip_thinking_for_model(target_model):
        base_payload.pop("thinking", None)

    base_payload["max_tokens"] = _resolve_native_message_max_tokens(base_payload, target_model)
    return base_payload


def _apply_native_message_reasoning(
    *,
    payload: dict[str, Any],
    model: str,
    reasoning_effort: str,
) -> None:
    if _is_claude_reasoning_model(model):
        output_config = payload.get("output_config")
        merged_output_config = dict(output_config) if isinstance(output_config, dict) else {}
        merged_output_config["effort"] = reasoning_effort
        payload["output_config"] = merged_output_config
        payload.pop("reasoning", None)
        return

    reasoning = payload.get("reasoning")
    merged_reasoning = dict(reasoning) if isinstance(reasoning, dict) else {}
    merged_reasoning["effort"] = reasoning_effort
    payload["reasoning"] = merged_reasoning
    payload.pop("output_config", None)


def _is_claude_reasoning_model(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized in {
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-opus-4-5",
    }


def _should_strip_thinking_for_model(model: str) -> bool:
    return not _is_claude_reasoning_model(model)


def _resolve_native_message_max_tokens(payload: dict[str, Any], model: str) -> int:
    parsed_value = _safe_int(payload.get("max_tokens"))
    if parsed_value <= 0:
        parsed_value = 4096

    model_limit = resolve_model_max_output_tokens(model)
    if model_limit is not None:
        return min(parsed_value, model_limit)
    return parsed_value


def _extract_native_message_path(payload: dict[str, Any]) -> str:
    query_string = payload.get("__proxy_query_string")
    if not isinstance(query_string, str):
        return "/messages"
    stripped = query_string.strip()
    if not stripped:
        return "/messages"
    return f"/messages?{stripped}"


def _extract_native_forward_headers(payload: dict[str, Any]) -> dict[str, str]:
    raw_headers = payload.get("__proxy_forward_headers")
    if not isinstance(raw_headers, dict):
        return {}
    headers: dict[str, str] = {}
    for key, value in raw_headers.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        stripped_key = key.strip()
        stripped_value = value.strip()
        if not stripped_key or not stripped_value:
            continue
        headers[stripped_key] = stripped_value
    return headers


def _is_retryable_upstream_error(error: UpstreamAPIError) -> bool:
    if error.status_code in {429, 500, 502, 503, 504}:
        return True
    payload = error.payload
    if isinstance(payload, dict):
        error_obj = payload.get("error")
        if isinstance(error_obj, dict):
            code = error_obj.get("code")
            if isinstance(code, str) and code.upper() in {"INTERNAL", "OVERLOADED", "RATE_LIMITED"}:
                return True
            message = error_obj.get("message")
            if isinstance(message, str) and "no available" in message.lower():
                return True
    return False


_RETRYABLE_UPSTREAM_STATUS_CODES = {429, 500, 502, 503, 504}


async def _post_json_with_retry(
    *,
    client: httpx.AsyncClient,
    path: str,
    payload: dict[str, Any],
    provider: str,
    stream: bool,
    raw_logger: RawIOLogger | None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            response = await client.post(path, json=payload, headers=headers)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            if attempt >= attempts:
                raise _wrap_transport_exception(
                    exc,
                    provider=provider,
                    path=path,
                    stream=stream,
                    raw_logger=raw_logger,
                ) from exc
            _log_transport_retry(
                raw_logger,
                provider=provider,
                path=path,
                stream=stream,
                attempt=attempt,
                attempts=attempts,
                exc=exc,
            )
            await asyncio.sleep(_retry_delay_seconds(attempt))
            continue

        if response.status_code in _RETRYABLE_UPSTREAM_STATUS_CODES and attempt < attempts:
            _log_response_retry(
                raw_logger,
                provider=provider,
                path=path,
                stream=stream,
                attempt=attempt,
                attempts=attempts,
                response=response,
            )
            await asyncio.sleep(_retry_delay_seconds(attempt, response=response))
            continue
        return response

    raise UpstreamAPIError(
        502,
        {
            "error": {
                "message": "Upstream request retry loop exited unexpectedly.",
                "type": "server_error",
                "param": None,
                "code": "unexpected_retry_exit",
            }
        },
    )


async def _send_request_with_retry(
    *,
    client: httpx.AsyncClient,
    request: httpx.Request,
    provider: str,
    path: str,
    stream: bool,
    raw_logger: RawIOLogger | None,
) -> httpx.Response:
    attempts = 3
    current_request = request
    for attempt in range(1, attempts + 1):
        try:
            response = await client.send(current_request, stream=True)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            if attempt >= attempts:
                raise _wrap_transport_exception(
                    exc,
                    provider=provider,
                    path=path,
                    stream=stream,
                    raw_logger=raw_logger,
                ) from exc
            _log_transport_retry(
                raw_logger,
                provider=provider,
                path=path,
                stream=stream,
                attempt=attempt,
                attempts=attempts,
                exc=exc,
            )
            await asyncio.sleep(_retry_delay_seconds(attempt))
            current_request = client.build_request(
                current_request.method,
                str(current_request.url),
                content=current_request.content,
                headers=current_request.headers,
            )
            continue

        if response.status_code in _RETRYABLE_UPSTREAM_STATUS_CODES and attempt < attempts:
            _log_response_retry(
                raw_logger,
                provider=provider,
                path=path,
                stream=stream,
                attempt=attempt,
                attempts=attempts,
                response=response,
            )
            await response.aclose()
            await asyncio.sleep(_retry_delay_seconds(attempt, response=response))
            current_request = client.build_request(
                current_request.method,
                str(current_request.url),
                content=current_request.content,
                headers=current_request.headers,
            )
            continue
        return response

    raise UpstreamAPIError(
        502,
        {
            "error": {
                "message": "Upstream streaming retry loop exited unexpectedly.",
                "type": "server_error",
                "param": None,
                "code": "unexpected_retry_exit",
            }
        },
    )


def _wrap_transport_exception(
    exc: httpx.HTTPError,
    *,
    provider: str,
    path: str,
    stream: bool,
    raw_logger: RawIOLogger | None,
) -> UpstreamAPIError:
    status_code = 504 if isinstance(exc, httpx.TimeoutException) else 502
    payload = {
        "error": {
            "message": f"Upstream transport error: {exc}",
            "type": "server_error",
            "param": None,
            "code": "upstream_timeout" if status_code == 504 else "upstream_connection_error",
        }
    }
    if raw_logger is not None:
        raw_logger.log(
            "upstream.error",
            {
                "provider": provider,
                "path": path,
                "stream": stream,
                "status_code": status_code,
                "error_type": type(exc).__name__,
                "error": payload,
            },
        )
    return UpstreamAPIError(status_code, payload)


def _log_transport_retry(
    raw_logger: RawIOLogger | None,
    *,
    provider: str,
    path: str,
    stream: bool,
    attempt: int,
    attempts: int,
    exc: httpx.HTTPError,
) -> None:
    if raw_logger is None:
        return
    raw_logger.log(
        "upstream.retry",
        {
            "provider": provider,
            "path": path,
            "stream": stream,
            "attempt": attempt,
            "attempts": attempts,
            "retrying": attempt < attempts,
            "error_type": type(exc).__name__,
            "message": str(exc),
        },
    )


def _log_response_retry(
    raw_logger: RawIOLogger | None,
    *,
    provider: str,
    path: str,
    stream: bool,
    attempt: int,
    attempts: int,
    response: httpx.Response,
) -> None:
    if raw_logger is None:
        return
    raw_logger.log(
        "upstream.retry",
        {
            "provider": provider,
            "path": path,
            "stream": stream,
            "attempt": attempt,
            "attempts": attempts,
            "retrying": attempt < attempts,
            "status_code": response.status_code,
        },
    )


def _build_response_failed_sse_lines(error: UpstreamAPIError) -> list[str]:
    error_obj = error.payload.get("error") if isinstance(error.payload, dict) else None
    if not isinstance(error_obj, dict):
        error_obj = {
            "message": "Upstream streaming request failed.",
            "type": "server_error",
            "param": None,
            "code": "upstream_stream_failed",
        }
    failure_event = {
        "type": "response.failed",
        "response": {
            "status": "failed",
            "error": error_obj,
        },
    }
    return [
        "event: response.failed",
        f"data: {json.dumps(failure_event, ensure_ascii=False)}",
        "",
        "data: [DONE]",
        "",
    ]


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


async def _collect_completed_response_from_sse(lines: AsyncIterator[str]) -> dict[str, Any]:
    last_response: dict[str, Any] | None = None
    async for payload in _iter_sse_json_payloads(lines):
        if payload == "[DONE]":
            break
        if not isinstance(payload, dict):
            continue
        payload_type = payload.get("type")
        if payload_type == "response.completed":
            response_obj = payload.get("response")
            if isinstance(response_obj, dict):
                return response_obj
        if payload_type == "response.failed":
            response_obj = payload.get("response")
            if isinstance(response_obj, dict):
                error_obj = response_obj.get("error")
                normalized_error = (
                    error_obj
                    if isinstance(error_obj, dict)
                    else {
                        "message": "Upstream streaming request failed.",
                        "type": "server_error",
                        "param": None,
                        "code": "upstream_stream_failed",
                    }
                )
                raise UpstreamAPIError(
                    _status_code_from_response_failed_error(normalized_error),
                    {"error": normalized_error},
                )
        if payload_type == "response.created":
            response_obj = payload.get("response")
            if isinstance(response_obj, dict):
                last_response = response_obj

    if last_response is not None:
        return last_response

    raise UpstreamAPIError(
        502,
        {
            "error": {
                "message": "Upstream streaming request completed without a final response payload.",
                "type": "server_error",
                "param": None,
                "code": "missing_stream_completion",
            }
        },
    )


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


def _status_code_from_response_failed_error(error_obj: dict[str, Any]) -> int:
    error_type = error_obj.get("type")
    if not isinstance(error_type, str):
        return 502

    normalized_type = error_type.strip().lower()
    if normalized_type == "invalid_request_error":
        return 400
    if normalized_type == "authentication_error":
        return 401
    if normalized_type == "permission_error":
        return 403
    if normalized_type == "not_found_error":
        return 404
    if normalized_type == "conflict_error":
        return 409
    if normalized_type == "unprocessable_entity_error":
        return 422
    if normalized_type == "rate_limit_error":
        return 429
    if normalized_type in {"server_error", "api_error"}:
        return 500
    return 502


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return 0
        try:
            return int(stripped)
        except ValueError:
            return 0
    return 0


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
    if "max_output_tokens" not in normalized:
        return normalized

    raw_value = normalized.get("max_output_tokens")
    if isinstance(raw_value, bool):
        normalized.pop("max_output_tokens", None)
        return normalized
    if isinstance(raw_value, int):
        value = raw_value
    elif isinstance(raw_value, float):
        value = int(raw_value)
    elif isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            normalized.pop("max_output_tokens", None)
            return normalized
        try:
            value = int(stripped)
        except ValueError:
            normalized.pop("max_output_tokens", None)
            return normalized
    else:
        normalized.pop("max_output_tokens", None)
        return normalized

    if value < 1:
        normalized.pop("max_output_tokens", None)
        return normalized
    if value > max_output_limit:
        value = max_output_limit

    normalized["max_output_tokens"] = value
    return normalized


def _is_invalid_argument_error(status_code: int, payload: Any) -> bool:
    if status_code != 400:
        return False
    if not isinstance(payload, dict):
        return False
    error_obj = payload.get("error")
    if not isinstance(error_obj, dict):
        return False
    code = error_obj.get("status")
    if isinstance(code, str) and code.upper() == "INVALID_ARGUMENT":
        return True
    message = error_obj.get("message")
    if isinstance(message, str) and "invalid argument" in message.lower():
        return True
    return False


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
