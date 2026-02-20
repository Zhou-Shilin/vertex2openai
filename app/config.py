"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return default


def _parse_csv(value: Optional[str]) -> Tuple[str, ...]:
    if not value:
        return ()
    parts = [item.strip() for item in value.split(",")]
    return tuple(item for item in parts if item)


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    log_level: str
    log_redact_body: bool
    request_body_log_max_chars: int
    cors_allowed_origins: Tuple[str, ...]
    google_api_base: str
    upstream_first_byte_timeout_seconds: int
    upstream_stream_read_timeout_seconds: int
    fallback_models: Tuple[str, ...]
    redis_url: str
    response_store_ttl_seconds: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        host=os.getenv("HOST", "0.0.0.0"),
        port=_parse_int(os.getenv("PORT"), 8080),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        log_redact_body=_parse_bool(os.getenv("LOG_REDACT_BODY"), True),
        request_body_log_max_chars=_parse_int(os.getenv("REQUEST_BODY_LOG_MAX_CHARS"), 512),
        cors_allowed_origins=_parse_csv(os.getenv("CORS_ALLOWED_ORIGINS")),
        google_api_base=os.getenv(
            "GOOGLE_API_BASE",
            "https://aiplatform.googleapis.com/v1",
        ).rstrip("/"),
        upstream_first_byte_timeout_seconds=_parse_int(
            os.getenv("UPSTREAM_FIRST_BYTE_TIMEOUT_SECONDS"),
            120,
        ),
        upstream_stream_read_timeout_seconds=_parse_int(
            os.getenv("UPSTREAM_STREAM_READ_TIMEOUT_SECONDS"),
            0,
        ),
        fallback_models=_parse_csv(
            os.getenv(
                "FALLBACK_MODELS",
                "gemini-3-flash-preview,gemini-2.5-flash,gemini-2.5-flash-lite,gemini-2.0-flash,gemini-2.0-flash-lite",
            )
        ),
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        response_store_ttl_seconds=_parse_int(os.getenv("RESPONSE_STORE_TTL_SECONDS"), 86400),
    )
