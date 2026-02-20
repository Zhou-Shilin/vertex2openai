"""OpenAI-compatible models endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.auth import extract_api_key
from app.clients.google_gemini import GoogleGeminiClient
from app.errors import APIError

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models")
async def list_models(request: Request) -> JSONResponse:
    api_key = extract_api_key(request)
    client = _get_gemini_client(request)
    try:
        models = await client.list_models(api_key)
    except APIError as exc:
        if exc.status_code != 404:
            raise
        models = []

    items: list[dict[str, Any]] = []
    for model in models:
        name = model.get("name")
        if not isinstance(name, str) or not name:
            continue
        model_id = _normalize_model_id(name)
        if not model_id:
            continue
        items.append(
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "google",
            }
        )

    if not items:
        fallback_models = tuple(getattr(request.app.state.settings, "fallback_models", ()) or ())
        for model_id in fallback_models:
            if not isinstance(model_id, str) or not model_id.strip():
                continue
            items.append(
                {
                    "id": model_id.strip(),
                    "object": "model",
                    "created": 0,
                    "owned_by": "google",
                }
            )

    items.sort(key=lambda item: item["id"])
    return JSONResponse({"object": "list", "data": items})


def _get_gemini_client(request: Request) -> GoogleGeminiClient:
    client = getattr(request.app.state, "gemini_client", None)
    if client is None:
        raise APIError(
            "Gemini client is not initialized.",
            500,
            error_type="server_error",
            code="service_unavailable",
        )
    return client


def _normalize_model_id(name: str) -> str:
    trimmed = name.strip().strip("/")
    if not trimmed:
        return ""
    if "/" not in trimmed:
        return trimmed
    return trimmed.split("/")[-1]
