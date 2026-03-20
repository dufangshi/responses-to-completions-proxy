from __future__ import annotations

import time

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.services.model_limits import resolve_model_limits

router = APIRouter()

def _resolve_limits(model_id: str) -> tuple[int, int]:
    limits = resolve_model_limits(model_id)
    return limits.context_window, limits.max_output_tokens


def _build_model_object(model_id: str) -> dict:
    context_window, max_tokens = _resolve_limits(model_id)
    now = int(time.time())
    return {
        "id": model_id,
        "object": "model",
        "created": now,
        "owned_by": "compat-proxy",
        "contextWindow": context_window,
        "maxTokens": max_tokens,
        "context_window": context_window,
        "max_output_tokens": max_tokens,
        "metadata": {
            "contextWindow": context_window,
            "maxTokens": max_tokens,
        },
    }


def _collect_model_ids(request: Request) -> list[str]:
    settings = request.app.state.settings
    model_ids: list[str] = []

    if settings.default_upstream_model:
        model_ids.append(settings.default_upstream_model)

    if settings.upstream_fallback_model:
        model_ids.append(settings.upstream_fallback_model)

    for model_id in settings.force_model_chain():
        if model_id:
            model_ids.append(model_id)

    deduped: list[str] = []
    seen: set[str] = set()
    for model_id in model_ids:
        key = model_id.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(model_id)
    return deduped


@router.get("/v1/models")
@router.get("/models")
async def list_models(request: Request) -> JSONResponse:
    model_ids = _collect_model_ids(request)
    data = [_build_model_object(model_id) for model_id in model_ids]
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"object": "list", "data": data},
    )


@router.get("/v1/models/{model_id}")
@router.get("/models/{model_id}")
async def get_model(model_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=_build_model_object(model_id),
    )
