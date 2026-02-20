"""Client for Google Gemini REST API via API key."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Optional

import httpx

from app.errors import APIError, upstream_status_to_api_error


class GoogleGeminiClient:
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        *,
        api_base: str,
        first_byte_timeout_seconds: int = 120,
        stream_read_timeout_seconds: int = 0,
    ) -> None:
        self._http = http_client
        self._api_base = api_base.rstrip("/")
        self._first_byte_timeout_seconds = first_byte_timeout_seconds
        self._stream_read_timeout_seconds = stream_read_timeout_seconds

    def _is_vertex_aiplatform_endpoint(self) -> bool:
        return "aiplatform.googleapis.com" in self._api_base.lower()

    def _normalize_model(self, model: str) -> str:
        normalized = model.strip()
        if not normalized:
            return normalized
        if normalized.startswith("/"):
            normalized = normalized[1:]

        if self._is_vertex_aiplatform_endpoint():
            if normalized.startswith(("projects/", "publishers/")):
                return normalized
            if normalized.startswith("models/"):
                normalized = normalized[len("models/") :]
            if "/" not in normalized:
                return f"publishers/google/models/{normalized}"
            return normalized

        if normalized.startswith("models/"):
            return normalized
        return f"models/{normalized}"

    async def generate_content(self, model: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        url = f"{self._api_base}/{self._normalize_model(model)}:generateContent"
        try:
            response = await self._http.post(url, params={"key": api_key}, json=payload)
        except httpx.TimeoutException as exc:
            raise APIError(
                "Upstream request timed out.",
                504,
                error_type="server_error",
                code="upstream_timeout",
            ) from exc
        except httpx.RequestError as exc:
            raise APIError(
                f"Failed to reach upstream service: {exc.__class__.__name__}.",
                502,
                error_type="server_error",
                code="upstream_connection_error",
            ) from exc

        if response.status_code >= 400:
            raise self._build_upstream_error(response.status_code, response.text)

        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise APIError(
                "Upstream returned invalid JSON.",
                502,
                error_type="server_error",
                code="upstream_invalid_json",
            ) from exc

    async def stream_generate_content(
        self,
        model: str,
        payload: dict[str, Any],
        api_key: str,
    ) -> AsyncIterator[dict[str, Any]]:
        url = f"{self._api_base}/{self._normalize_model(model)}:streamGenerateContent"
        params = {"key": api_key, "alt": "sse"}
        try:
            async with self._http.stream("POST", url, params=params, json=payload) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="ignore")
                    raise self._build_upstream_error(response.status_code, body)

                lines = response.aiter_lines()
                first_payload = await self._read_next_sse_payload(
                    lines,
                    first_byte_timeout_seconds=self._first_byte_timeout_seconds,
                )
                if first_payload is None:
                    return
                yield first_payload

                while True:
                    timeout_seconds: Optional[int] = None
                    if self._stream_read_timeout_seconds > 0:
                        timeout_seconds = self._stream_read_timeout_seconds
                    next_payload = await self._read_next_sse_payload(
                        lines,
                        first_byte_timeout_seconds=timeout_seconds,
                    )
                    if next_payload is None:
                        break
                    yield next_payload
        except asyncio.TimeoutError as exc:
            raise APIError(
                "Upstream first byte timeout.",
                504,
                error_type="server_error",
                code="upstream_timeout",
            ) from exc
        except httpx.RequestError as exc:
            raise APIError(
                f"Streaming connection to upstream failed: {exc.__class__.__name__}.",
                502,
                error_type="server_error",
                code="upstream_connection_error",
            ) from exc

    async def list_models(self, api_key: str) -> list[dict[str, Any]]:
        list_paths = ["models"]
        if self._is_vertex_aiplatform_endpoint():
            list_paths = ["publishers/google/models", "models"]

        models: list[dict[str, Any]] = []
        last_error: Optional[APIError] = None

        for path in list_paths:
            url = f"{self._api_base}/{path}"
            page_token: Optional[str] = None
            path_models: list[dict[str, Any]] = []
            for _ in range(20):
                params: dict[str, Any] = {"key": api_key, "pageSize": 100}
                if page_token:
                    params["pageToken"] = page_token
                try:
                    response = await self._http.get(url, params=params)
                except httpx.TimeoutException as exc:
                    raise APIError(
                        "Upstream models request timed out.",
                        504,
                        error_type="server_error",
                        code="upstream_timeout",
                    ) from exc
                except httpx.RequestError as exc:
                    raise APIError(
                        f"Failed to reach upstream service: {exc.__class__.__name__}.",
                        502,
                        error_type="server_error",
                        code="upstream_connection_error",
                    ) from exc

                if response.status_code >= 400:
                    candidate_error = self._build_upstream_error(response.status_code, response.text)
                    if response.status_code == 404:
                        last_error = candidate_error
                        path_models = []
                        break
                    raise candidate_error

                try:
                    payload = response.json()
                except json.JSONDecodeError as exc:
                    raise APIError(
                        "Upstream returned invalid JSON for models list.",
                        502,
                        error_type="server_error",
                        code="upstream_invalid_json",
                    ) from exc

                page_models = payload.get("models")
                if isinstance(page_models, list):
                    path_models.extend(item for item in page_models if isinstance(item, dict))
                page_token = payload.get("nextPageToken")
                if not page_token:
                    break

            if path_models:
                models.extend(path_models)
                break

        if models:
            return models
        if last_error is not None:
            raise last_error
        return models

    async def count_tokens(self, model: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        url = f"{self._api_base}/{self._normalize_model(model)}:countTokens"
        try:
            response = await self._http.post(url, params={"key": api_key}, json=payload)
        except httpx.TimeoutException as exc:
            raise APIError(
                "Upstream request timed out.",
                504,
                error_type="server_error",
                code="upstream_timeout",
            ) from exc
        except httpx.RequestError as exc:
            raise APIError(
                f"Failed to reach upstream service: {exc.__class__.__name__}.",
                502,
                error_type="server_error",
                code="upstream_connection_error",
            ) from exc

        if response.status_code >= 400:
            raise self._build_upstream_error(response.status_code, response.text)

        try:
            payload_json = response.json()
        except json.JSONDecodeError as exc:
            raise APIError(
                "Upstream returned invalid JSON.",
                502,
                error_type="server_error",
                code="upstream_invalid_json",
            ) from exc
        if isinstance(payload_json, dict):
            return payload_json
        raise APIError(
            "Upstream returned invalid token-count payload.",
            502,
            error_type="server_error",
            code="upstream_invalid_json",
        )

    async def _read_next_sse_payload(
        self,
        line_iterator: Any,
        *,
        first_byte_timeout_seconds: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        async def _inner() -> Optional[dict[str, Any]]:
            async for line in line_iterator:
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    return None
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    continue
            return None

        if first_byte_timeout_seconds is None or first_byte_timeout_seconds <= 0:
            return await _inner()
        return await asyncio.wait_for(_inner(), timeout=first_byte_timeout_seconds)

    def _build_upstream_error(self, status_code: int, body: str) -> APIError:
        message = "Upstream request failed."
        try:
            payload = json.loads(body)
            if isinstance(payload, dict):
                upstream_error = payload.get("error")
                if isinstance(upstream_error, dict):
                    message = str(upstream_error.get("message") or message)
                elif payload.get("message"):
                    message = str(payload["message"])
        except json.JSONDecodeError:
            if body.strip():
                message = body.strip()[:300]
        return upstream_status_to_api_error(status_code, message)
