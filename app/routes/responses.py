"""OpenAI Responses-compatible endpoints."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.auth import extract_api_key
from app.clients.google_gemini import GoogleGeminiClient
from app.errors import APIError
from app.store.redis_store import RedisResponseStore
from app.translate.google_to_openai import (
    StreamAggregateState,
    build_stored_record,
    make_message_output_item,
    make_response_object,
    new_response_id,
    sse_event,
    translate_google_response_to_openai,
)
from app.translate.openai_to_google import translate_openai_request_to_google

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["responses"])


@router.post("/responses")
async def create_response(request: Request) -> Response:
    payload = await _read_json_payload(request)
    _validate_create_payload(payload)

    model = str(payload["model"])
    stream = payload.get("stream", False)
    store = payload.get("store", True)
    previous_response_id = payload.get("previous_response_id")
    api_key = extract_api_key(request)

    response_store = _get_response_store(request)
    previous_response: Optional[dict[str, Any]] = None
    if previous_response_id is not None:
        if not isinstance(previous_response_id, str) or not previous_response_id.strip():
            raise APIError(
                "`previous_response_id` must be a non-empty string.",
                400,
                param="previous_response_id",
                error_type="invalid_request_error",
                code="invalid_type",
            )
        record = await response_store.get(previous_response_id)
        if not record:
            raise APIError(
                f"Response `{previous_response_id}` was not found.",
                404,
                param="previous_response_id",
                error_type="not_found_error",
                code="not_found",
            )
        previous_response = record.get("response")

    translated = translate_openai_request_to_google(payload, previous_response=previous_response)
    if translated.warnings:
        logger.warning(
            "Request translated with degradations. response_id=pending warnings=%s",
            translated.warnings,
        )

    gemini = _get_gemini_client(request)
    response_id = new_response_id()

    if stream:
        return StreamingResponse(
            _stream_create_response(
                request=request,
                gemini=gemini,
                response_store=response_store,
                api_key=api_key,
                model=model,
                translated_payload=translated.payload,
                original_request_payload=payload,
                response_id=response_id,
                persist=store,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    google_response = await gemini.generate_content(model, translated.payload, api_key)
    openai_response = translate_google_response_to_openai(
        google_payload=google_response,
        model=model,
        response_id=response_id,
    )
    if store:
        settings = request.app.state.settings
        record = build_stored_record(
            response_id=response_id,
            request_payload=payload,
            response_payload=openai_response,
            model=model,
            ttl_seconds=settings.response_store_ttl_seconds,
        )
        await response_store.save(record)
    return JSONResponse(openai_response)


@router.get("/responses/{response_id}")
async def get_response(response_id: str, request: Request) -> JSONResponse:
    response_store = _get_response_store(request)
    record = await response_store.get(response_id)
    if not record:
        raise APIError(
            f"Response `{response_id}` was not found.",
            404,
            param="response_id",
            error_type="not_found_error",
            code="not_found",
        )
    response_payload = record.get("response")
    if not isinstance(response_payload, dict):
        raise APIError(
            f"Stored response `{response_id}` is corrupted.",
            500,
            error_type="server_error",
            code="storage_corruption",
        )
    return JSONResponse(response_payload)


@router.delete("/responses/{response_id}")
async def delete_response(response_id: str, request: Request) -> JSONResponse:
    response_store = _get_response_store(request)
    deleted = await response_store.delete(response_id)
    if not deleted:
        raise APIError(
            f"Response `{response_id}` was not found.",
            404,
            param="response_id",
            error_type="not_found_error",
            code="not_found",
        )
    return JSONResponse(
        {
            "id": response_id,
            "object": "response.deleted",
            "deleted": True,
        }
    )


async def _stream_create_response(
    *,
    request: Request,
    gemini: GoogleGeminiClient,
    response_store: RedisResponseStore,
    api_key: str,
    model: str,
    translated_payload: dict[str, Any],
    original_request_payload: dict[str, Any],
    response_id: str,
    persist: bool,
) -> AsyncIterator[str]:
    state = StreamAggregateState(response_id=response_id, model=model)
    created_response = make_response_object(
        response_id=response_id,
        model=model,
        output=[],
        usage=state.usage,
        status="in_progress",
        created_at=state.created_at,
    )
    yield sse_event("response.created", {"response": created_response})
    yield sse_event(
        "response.in_progress",
        {"response": {"id": response_id, "object": "response", "status": "in_progress"}},
    )

    try:
        async for chunk in gemini.stream_generate_content(model, translated_payload, api_key):
            if not isinstance(chunk, dict):
                continue

            state.update_usage(chunk.get("usageMetadata"))

            full_text, function_calls = _extract_stream_elements(chunk)
            delta = state.set_text(full_text)
            if delta:
                if not state.text_item_emitted:
                    state.text_item_emitted = True
                    item = make_message_output_item("", item_id=state.text_item_id)
                    yield sse_event(
                        "response.output_item.added",
                        {
                            "response_id": response_id,
                            "output_index": 0,
                            "item": item,
                        },
                    )
                yield sse_event(
                    "response.output_text.delta",
                    {
                        "response_id": response_id,
                        "item_id": state.text_item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": delta,
                    },
                )

            for fc in function_calls:
                key = fc["key"]
                name = fc["name"]
                arguments = fc["arguments"]
                previous_arguments = ""
                if key in state.function_calls:
                    previous_arguments = state.function_calls[key]["arguments"]

                item, is_new = state.register_function_call(key=key, name=name, arguments=arguments)
                if is_new:
                    output_index = len(state.function_calls) - 1 + (1 if state.text_value else 0)
                    yield sse_event(
                        "response.output_item.added",
                        {
                            "response_id": response_id,
                            "output_index": output_index,
                            "item": item,
                        },
                    )
                else:
                    delta_args = _diff_arguments(previous_arguments, arguments)
                    if delta_args:
                        yield sse_event(
                            "response.function_call_arguments.delta",
                            {
                                "response_id": response_id,
                                "item_id": item["id"],
                                "call_id": item["call_id"],
                                "delta": delta_args,
                            },
                        )

        final_response = state.build_final_response()
        if persist:
            settings = request.app.state.settings
            record = build_stored_record(
                response_id=response_id,
                request_payload=original_request_payload,
                response_payload=final_response,
                model=model,
                ttl_seconds=settings.response_store_ttl_seconds,
            )
            await response_store.save(record)
        yield sse_event("response.completed", {"response": final_response})
    except APIError as exc:
        yield sse_event("response.error", {"response_id": response_id, "error": exc.to_payload()["error"]})
    except Exception:
        logger.exception("Streaming failed. response_id=%s", response_id)
        api_error = APIError(
            "Streaming failed.",
            502,
            error_type="server_error",
            code="stream_error",
        )
        yield sse_event("response.error", {"response_id": response_id, "error": api_error.to_payload()["error"]})
    finally:
        yield "data: [DONE]\n\n"


def _extract_stream_elements(chunk: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    text_chunks: list[str] = []
    function_calls: list[dict[str, str]] = []

    candidates = chunk.get("candidates")
    if not isinstance(candidates, list):
        return "", function_calls

    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part_index, part in enumerate(parts):
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                text_chunks.append(text)
            function_call = part.get("functionCall")
            if isinstance(function_call, dict):
                name = function_call.get("name")
                if not isinstance(name, str) or not name:
                    continue
                args = function_call.get("args")
                key = str(function_call.get("id") or f"{candidate_index}:{part_index}:{name}")
                function_calls.append(
                    {
                        "key": key,
                        "name": name,
                        "arguments": json.dumps(args if args is not None else {}, ensure_ascii=False, separators=(",", ":")),
                    }
                )

    return "".join(text_chunks), function_calls


def _diff_arguments(previous_value: str, new_value: str) -> str:
    if new_value.startswith(previous_value):
        return new_value[len(previous_value) :]
    if previous_value.startswith(new_value):
        return ""
    return new_value


async def _read_json_payload(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise APIError(
            "Request body must be valid JSON.",
            400,
            error_type="invalid_request_error",
            code="invalid_json",
        ) from exc

    if not isinstance(payload, dict):
        raise APIError(
            "Request body must be a JSON object.",
            400,
            error_type="invalid_request_error",
            code="invalid_json",
        )
    return payload


def _validate_create_payload(payload: dict[str, Any]) -> None:
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise APIError(
            "Missing required field `model`.",
            400,
            param="model",
            error_type="invalid_request_error",
            code="invalid_model",
        )

    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise APIError(
            "`stream` must be a boolean.",
            400,
            param="stream",
            error_type="invalid_request_error",
            code="invalid_type",
        )

    store = payload.get("store", True)
    if not isinstance(store, bool):
        raise APIError(
            "`store` must be a boolean.",
            400,
            param="store",
            error_type="invalid_request_error",
            code="invalid_type",
        )


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


def _get_response_store(request: Request) -> RedisResponseStore:
    store = getattr(request.app.state, "response_store", None)
    if store is None:
        raise APIError(
            "Response store is not initialized.",
            500,
            error_type="server_error",
            code="service_unavailable",
        )
    return store
