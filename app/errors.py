"""OpenAI-style error handling."""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Normalized API error that renders as OpenAI-style payload."""

    def __init__(
        self,
        message: str,
        status_code: int,
        *,
        error_type: str = "invalid_request_error",
        param: Optional[str] = None,
        code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.param = param
        self.code = code

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


def upstream_status_to_api_error(status_code: int, message: str) -> APIError:
    if status_code == 400:
        return APIError(
            message,
            400,
            error_type="invalid_request_error",
            code="bad_request",
        )
    if status_code in {401, 403}:
        return APIError(
            message,
            status_code,
            error_type="authentication_error",
            code="invalid_api_key",
        )
    if status_code == 404:
        return APIError(
            message,
            404,
            error_type="not_found_error",
            code="not_found",
        )
    if status_code == 429:
        return APIError(
            message,
            429,
            error_type="rate_limit_error",
            code="rate_limit_exceeded",
        )
    if status_code == 408:
        return APIError(
            "Upstream request timed out.",
            504,
            error_type="server_error",
            code="upstream_timeout",
        )
    if status_code >= 500:
        return APIError(
            "Upstream service failed.",
            502,
            error_type="server_error",
            code="upstream_error",
        )
    return APIError(
        "Upstream request failed.",
        502,
        error_type="server_error",
        code="upstream_error",
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def handle_api_error(_: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        message = "; ".join(err.get("msg", "Validation error") for err in exc.errors())
        api_error = APIError(
            message or "Request validation failed.",
            400,
            error_type="invalid_request_error",
            code="validation_error",
        )
        return JSONResponse(status_code=api_error.status_code, content=api_error.to_payload())

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        status = exc.status_code if 400 <= exc.status_code < 600 else 500
        if status == 404:
            api_error = APIError(
                "Resource not found.",
                404,
                error_type="not_found_error",
                code="not_found",
            )
        else:
            api_error = APIError(
                str(exc.detail) if exc.detail else "HTTP error.",
                status,
                error_type="invalid_request_error" if status < 500 else "server_error",
                code="http_error",
            )
        return JSONResponse(status_code=api_error.status_code, content=api_error.to_payload())

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception("Unhandled exception. request_id=%s", request_id, exc_info=exc)
        api_error = APIError(
            "Internal server error.",
            500,
            error_type="server_error",
            code="internal_error",
        )
        return JSONResponse(status_code=api_error.status_code, content=api_error.to_payload())
