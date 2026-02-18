"""Authentication helpers."""

from __future__ import annotations

from typing import Optional

from fastapi import Request

from app.errors import APIError


def redact_api_key(api_key: Optional[str]) -> str:
    if not api_key:
        return "<missing>"
    trimmed = api_key.strip()
    if len(trimmed) <= 6:
        return "*" * len(trimmed)
    return f"{trimmed[:4]}...{trimmed[-2:]}"


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        raise APIError(
            "Authorization header must use Bearer scheme.",
            401,
            error_type="authentication_error",
            code="invalid_api_key",
        )
    token = token.strip()
    if not token:
        raise APIError(
            "Bearer token is empty.",
            401,
            error_type="authentication_error",
            code="invalid_api_key",
        )
    return token


def extract_api_key(request: Request) -> str:
    """
    Prefer Bearer token over query string key.
    Accept either:
    - Authorization: Bearer <key>
    - ?api_key=<key>
    """

    bearer = _extract_bearer_token(request.headers.get("authorization"))
    if bearer:
        return bearer

    query_key = request.query_params.get("api_key")
    if query_key and query_key.strip():
        return query_key.strip()

    raise APIError(
        "Missing API key. Provide Authorization: Bearer <key> or ?api_key=<key>.",
        401,
        error_type="authentication_error",
        code="invalid_api_key",
    )


def extract_api_key_for_logging(request: Request) -> str:
    try:
        bearer = _extract_bearer_token(request.headers.get("authorization"))
    except APIError:
        bearer = None
    if bearer:
        return redact_api_key(bearer)
    query_key = request.query_params.get("api_key")
    return redact_api_key(query_key)
