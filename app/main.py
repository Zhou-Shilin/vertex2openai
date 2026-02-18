"""FastAPI application entrypoint."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.auth import extract_api_key_for_logging
from app.clients.google_gemini import GoogleGeminiClient
from app.config import Settings, get_settings
from app.errors import install_error_handlers
from app.routes.models import router as models_router
from app.routes.responses import router as responses_router
from app.store.redis_store import RedisResponseStore


def create_app(
    *,
    settings: Optional[Settings] = None,
    injected_gemini_client: Optional[GoogleGeminiClient] = None,
    injected_response_store: Optional[RedisResponseStore] = None,
) -> FastAPI:
    cfg = settings or get_settings()
    _configure_logging(cfg.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = cfg
        app.state.gemini_client = None
        app.state.response_store = None
        app.state.http_client = None
        app.state._injected_gemini_client = injected_gemini_client
        app.state._injected_response_store = injected_response_store
        app.state._owns_http_client = False
        app.state._owns_response_store = False

        if app.state._injected_gemini_client is not None:
            app.state.gemini_client = app.state._injected_gemini_client
        else:
            timeout = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=60.0)
            app.state.http_client = httpx.AsyncClient(timeout=timeout)
            app.state._owns_http_client = True
            app.state.gemini_client = GoogleGeminiClient(
                app.state.http_client,
                api_base=cfg.google_api_base,
                first_byte_timeout_seconds=cfg.upstream_first_byte_timeout_seconds,
                stream_read_timeout_seconds=cfg.upstream_stream_read_timeout_seconds,
            )

        if app.state._injected_response_store is not None:
            app.state.response_store = app.state._injected_response_store
        else:
            store = RedisResponseStore(cfg.redis_url, cfg.response_store_ttl_seconds)
            await store.connect()
            app.state.response_store = store
            app.state._owns_response_store = True

        try:
            yield
        finally:
            if app.state._owns_response_store and app.state.response_store is not None:
                await app.state.response_store.close()
                app.state.response_store = None
            if app.state._owns_http_client and app.state.http_client is not None:
                await app.state.http_client.aclose()
                app.state.http_client = None

    app = FastAPI(title="Vertex2OpenAI", version="1.0.0", lifespan=lifespan)

    if cfg.cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cfg.cors_allowed_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_observability_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex}"
        request.state.request_id = request_id
        started = time.time()

        body_preview = "<redacted>"
        if not cfg.log_redact_body:
            body_bytes = await request.body()
            body_text = body_bytes.decode("utf-8", errors="ignore")
            body_preview = body_text[: cfg.request_body_log_max_chars]

            async def receive() -> dict[str, Any]:
                return {"type": "http.request", "body": body_bytes, "more_body": False}

            request = Request(request.scope, receive)

        api_key_for_log = extract_api_key_for_logging(request)
        logging.getLogger("vertex2openai.request").info(
            "request_id=%s method=%s path=%s api_key=%s body=%s",
            request_id,
            request.method,
            request.url.path,
            api_key_for_log,
            body_preview,
        )

        response = await call_next(request)
        duration_ms = int((time.time() - started) * 1000)
        response.headers["x-request-id"] = request_id
        logging.getLogger("vertex2openai.request").info(
            "request_id=%s status=%s duration_ms=%s",
            request_id,
            response.status_code,
            duration_ms,
        )
        return response

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        payload = {
            "status": "ok",
            "service": "vertex2openai",
            "version": "1.0.0",
            "timestamp": int(time.time()),
        }
        return JSONResponse(payload)

    @app.get("/")
    async def root() -> JSONResponse:
        return JSONResponse(
            {
                "name": "vertex2openai",
                "message": "OpenAI-compatible Responses API proxy for Google Gemini",
                "endpoints": ["/v1/responses", "/v1/models"],
            }
        )

    app.include_router(models_router)
    app.include_router(responses_router)
    install_error_handlers(app)
    return app


def _configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


app = create_app()
