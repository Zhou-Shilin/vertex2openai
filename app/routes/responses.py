"""OpenAI Responses-compatible endpoints."""

from __future__ import annotations

import asyncio
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
    background = payload.get("background", False)
    api_key = extract_api_key(request)

    response_store = _get_response_store(request)
    previous_response = await _load_previous_response_if_needed(payload, response_store)
    max_tool_calls = payload.get("max_tool_calls")

    translated = translate_openai_request_to_google(payload, previous_response=previous_response)
    if translated.warnings:
        logger.warning(
            "Request translated with degradations. response_id=pending warnings=%s",
            translated.warnings,
        )
    response_extra_fields = dict(translated.passthrough_fields)

    gemini = _get_gemini_client(request)
    response_id = new_response_id()

    if background:
        if stream:
            raise APIError(
                "`background=true` cannot be used with `stream=true`.",
                400,
                param="background",
                error_type="invalid_request_error",
                code="invalid_type",
            )
        if not store:
            raise APIError(
                "`background=true` requires `store=true` so result can be retrieved later.",
                400,
                param="store",
                error_type="invalid_request_error",
                code="invalid_type",
            )
        settings = request.app.state.settings
        queued_response = make_response_object(
            response_id=response_id,
            model=model,
            output=[],
            status="queued",
            extra_fields=response_extra_fields,
        )
        record = build_stored_record(
            response_id=response_id,
            request_payload=payload,
            response_payload=queued_response,
            model=model,
            ttl_seconds=settings.response_store_ttl_seconds,
        )
        await response_store.save(record)

        background_tasks = _get_background_tasks(request)
        task = asyncio.create_task(
            _run_background_response(
                request=request,
                gemini=gemini,
                response_store=response_store,
                api_key=api_key,
                model=model,
                translated_payload=translated.payload,
                original_request_payload=payload,
                response_id=response_id,
                response_extra_fields=response_extra_fields,
                max_tool_calls=max_tool_calls if isinstance(max_tool_calls, int) else None,
            )
        )
        background_tasks[response_id] = task
        return JSONResponse(queued_response)

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
                response_extra_fields=response_extra_fields,
                max_tool_calls=max_tool_calls if isinstance(max_tool_calls, int) else None,
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
        extra_fields=response_extra_fields,
    )
    openai_response = _apply_max_tool_calls_limit(openai_response, max_tool_calls)
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


@router.post("/responses/input_tokens")
async def count_response_input_tokens(request: Request) -> JSONResponse:
    payload = await _read_json_payload(request)
    _validate_model_field(payload)
    model = str(payload["model"])
    api_key = extract_api_key(request)
    response_store = _get_response_store(request)
    previous_response = await _load_previous_response_if_needed(payload, response_store)

    translated = translate_openai_request_to_google(payload, previous_response=previous_response)
    gemini = _get_gemini_client(request)
    count_payload = _build_count_tokens_payload(translated.payload)
    upstream = await gemini.count_tokens(model, count_payload, api_key)

    raw_total = upstream.get("totalTokens")
    input_tokens = int(raw_total) if isinstance(raw_total, (int, float)) else 0
    return JSONResponse(
        {
            "object": "response.input_tokens",
            "model": model,
            "input_tokens": input_tokens,
            "total_tokens": input_tokens,
        }
    )


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


@router.get("/responses/{response_id}/input_items")
async def list_response_input_items(response_id: str, request: Request) -> JSONResponse:
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
    request_payload = record.get("request")
    if not isinstance(request_payload, dict):
        raise APIError(
            f"Stored request for `{response_id}` is corrupted.",
            500,
            error_type="server_error",
            code="storage_corruption",
        )

    items = _input_items_from_request(request_payload)
    first_id = items[0]["id"] if items else None
    last_id = items[-1]["id"] if items else None
    return JSONResponse(
        {
            "object": "list",
            "data": items,
            "first_id": first_id,
            "last_id": last_id,
            "has_more": False,
        }
    )


@router.post("/responses/{response_id}/cancel")
async def cancel_response(response_id: str, request: Request) -> JSONResponse:
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

    background_tasks = _get_background_tasks(request)
    task = background_tasks.get(response_id)
    if task and not task.done():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    refreshed = await response_store.get(response_id)
    if refreshed and isinstance(refreshed.get("response"), dict):
        return JSONResponse(refreshed["response"])

    return JSONResponse(response_payload)


@router.post("/responses/{response_id}/compact")
async def compact_response(response_id: str, request: Request) -> JSONResponse:
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
    response_extra_fields: dict[str, Any],
    max_tool_calls: Optional[int],
) -> AsyncIterator[str]:
    state = StreamAggregateState(response_id=response_id, model=model, extra_fields=response_extra_fields)
    sequence_number = 0
    max_tool_calls_exceeded = False

    def emit(event_type: str, payload: dict[str, Any]) -> str:
        nonlocal sequence_number
        sequence_number += 1
        extended_payload = dict(payload)
        extended_payload["sequence_number"] = sequence_number
        return sse_event(event_type, extended_payload)

    created_response = make_response_object(
        response_id=response_id,
        model=model,
        output=[],
        usage=state.usage,
        status="in_progress",
        created_at=state.created_at,
        extra_fields=response_extra_fields,
    )
    yield emit("response.created", {"response": created_response})
    yield emit(
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
                    yield emit(
                        "response.output_item.added",
                        {
                            "response_id": response_id,
                            "output_index": 0,
                            "item": item,
                        },
                    )
                    yield emit(
                        "response.content_part.added",
                        {
                            "response_id": response_id,
                            "item_id": state.text_item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": ""},
                        },
                    )
                yield emit(
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
                if (
                    isinstance(max_tool_calls, int)
                    and key not in state.function_calls
                    and len(state.function_calls) >= max_tool_calls
                ):
                    max_tool_calls_exceeded = True
                    continue
                previous_arguments = ""
                if key in state.function_calls:
                    previous_arguments = state.function_calls[key]["arguments"]

                item, is_new = state.register_function_call(key=key, name=name, arguments=arguments)
                if is_new:
                    output_index = len(state.function_calls) - 1 + (1 if state.text_value else 0)
                    yield emit(
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
                        yield emit(
                            "response.function_call_arguments.delta",
                            {
                                "response_id": response_id,
                                "item_id": item["id"],
                                "call_id": item["call_id"],
                                "delta": delta_args,
                            },
                        )

        final_response = state.build_final_response()
        if max_tool_calls_exceeded:
            final_response = _apply_max_tool_calls_limit(final_response, max_tool_calls)
        if state.text_item_emitted:
            message_item = make_message_output_item(state.text_value, item_id=state.text_item_id)
            yield emit(
                "response.output_text.done",
                {
                    "response_id": response_id,
                    "item_id": state.text_item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": state.text_value,
                },
            )
            yield emit(
                "response.content_part.done",
                {
                    "response_id": response_id,
                    "item_id": state.text_item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": state.text_value},
                },
            )
            yield emit(
                "response.output_item.done",
                {
                    "response_id": response_id,
                    "output_index": 0,
                    "item": message_item,
                },
            )

        function_offset = 1 if state.text_item_emitted else 0
        for index, item in enumerate(state.function_calls.values()):
            yield emit(
                "response.function_call_arguments.done",
                {
                    "response_id": response_id,
                    "item_id": item["id"],
                    "call_id": item["call_id"],
                    "arguments": item["arguments"],
                },
            )
            yield emit(
                "response.output_item.done",
                {
                    "response_id": response_id,
                    "output_index": function_offset + index,
                    "item": item,
                },
            )

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
        yield emit("response.completed", {"response": final_response})
    except APIError as exc:
        error_payload = exc.to_payload()["error"]
        yield emit("response.failed", {"response_id": response_id, "error": error_payload})
        yield emit("error", {"response_id": response_id, "error": error_payload})
    except Exception:
        logger.exception("Streaming failed. response_id=%s", response_id)
        api_error = APIError(
            "Streaming failed.",
            502,
            error_type="server_error",
            code="stream_error",
        )
        error_payload = api_error.to_payload()["error"]
        yield emit("response.failed", {"response_id": response_id, "error": error_payload})
        yield emit("error", {"response_id": response_id, "error": error_payload})
    finally:
        yield "data: [DONE]\n\n"


async def _run_background_response(
    *,
    request: Request,
    gemini: GoogleGeminiClient,
    response_store: RedisResponseStore,
    api_key: str,
    model: str,
    translated_payload: dict[str, Any],
    original_request_payload: dict[str, Any],
    response_id: str,
    response_extra_fields: dict[str, Any],
    max_tool_calls: Optional[int],
) -> None:
    settings = request.app.state.settings
    background_tasks = _get_background_tasks(request)
    try:
        in_progress = make_response_object(
            response_id=response_id,
            model=model,
            output=[],
            status="in_progress",
            extra_fields=response_extra_fields,
        )
        await response_store.save(
            build_stored_record(
                response_id=response_id,
                request_payload=original_request_payload,
                response_payload=in_progress,
                model=model,
                ttl_seconds=settings.response_store_ttl_seconds,
            )
        )

        google_response = await gemini.generate_content(model, translated_payload, api_key)
        final_response = translate_google_response_to_openai(
            google_payload=google_response,
            model=model,
            response_id=response_id,
            extra_fields=response_extra_fields,
        )
        final_response = _apply_max_tool_calls_limit(final_response, max_tool_calls)
        await response_store.save(
            build_stored_record(
                response_id=response_id,
                request_payload=original_request_payload,
                response_payload=final_response,
                model=model,
                ttl_seconds=settings.response_store_ttl_seconds,
            )
        )
    except asyncio.CancelledError:
        cancelled_response = make_response_object(
            response_id=response_id,
            model=model,
            output=[],
            status="incomplete",
            incomplete_details={"reason": "cancelled"},
            extra_fields=response_extra_fields,
        )
        await response_store.save(
            build_stored_record(
                response_id=response_id,
                request_payload=original_request_payload,
                response_payload=cancelled_response,
                model=model,
                ttl_seconds=settings.response_store_ttl_seconds,
            )
        )
        raise
    except APIError as exc:
        failed_response = make_response_object(
            response_id=response_id,
            model=model,
            output=[],
            status="failed",
            incomplete_details={"reason": "error"},
            error=exc.to_payload()["error"],
            extra_fields=response_extra_fields,
        )
        await response_store.save(
            build_stored_record(
                response_id=response_id,
                request_payload=original_request_payload,
                response_payload=failed_response,
                model=model,
                ttl_seconds=settings.response_store_ttl_seconds,
            )
        )
    except Exception:
        logger.exception("Background response generation failed. response_id=%s", response_id)
        api_error = APIError(
            "Background response generation failed.",
            502,
            error_type="server_error",
            code="background_error",
        )
        failed_response = make_response_object(
            response_id=response_id,
            model=model,
            output=[],
            status="failed",
            incomplete_details={"reason": "error"},
            error=api_error.to_payload()["error"],
            extra_fields=response_extra_fields,
        )
        await response_store.save(
            build_stored_record(
                response_id=response_id,
                request_payload=original_request_payload,
                response_payload=failed_response,
                model=model,
                ttl_seconds=settings.response_store_ttl_seconds,
            )
        )
    finally:
        background_tasks.pop(response_id, None)


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
    _validate_model_field(payload)

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

    background = payload.get("background", False)
    if not isinstance(background, bool):
        raise APIError(
            "`background` must be a boolean.",
            400,
            param="background",
            error_type="invalid_request_error",
            code="invalid_type",
        )


def _validate_model_field(payload: dict[str, Any]) -> None:
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise APIError(
            "Missing required field `model`.",
            400,
            param="model",
            error_type="invalid_request_error",
            code="invalid_model",
        )


def _input_items_from_request(request_payload: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"in_{counter}"

    instructions = request_payload.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        result.append(
            {
                "id": next_id(),
                "type": "message",
                "role": "system",
                "status": "completed",
                "content": [{"type": "input_text", "text": instructions}],
            }
        )

    input_value = request_payload.get("input")
    if isinstance(input_value, str):
        result.append(
            {
                "id": next_id(),
                "type": "message",
                "role": "user",
                "status": "completed",
                "content": [{"type": "input_text", "text": input_value}],
            }
        )
        return result

    if isinstance(input_value, list):
        for item in input_value:
            if isinstance(item, str):
                result.append(
                    {
                        "id": next_id(),
                        "type": "message",
                        "role": "user",
                        "status": "completed",
                        "content": [{"type": "input_text", "text": item}],
                    }
                )
                continue
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            normalized.setdefault("id", next_id())
            normalized.setdefault("status", "completed")
            result.append(normalized)
    return result


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


def _get_background_tasks(request: Request) -> dict[str, asyncio.Task[Any]]:
    tasks = getattr(request.app.state, "background_tasks", None)
    if not isinstance(tasks, dict):
        tasks = {}
        request.app.state.background_tasks = tasks
    return tasks


async def _load_previous_response_if_needed(
    payload: dict[str, Any],
    response_store: RedisResponseStore,
) -> Optional[dict[str, Any]]:
    previous_response_id = payload.get("previous_response_id")
    if previous_response_id is None:
        return None
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
    response_payload = record.get("response")
    if not isinstance(response_payload, dict):
        raise APIError(
            f"Stored response `{previous_response_id}` is corrupted.",
            500,
            error_type="server_error",
            code="storage_corruption",
        )
    return response_payload


def _build_count_tokens_payload(translated_payload: dict[str, Any]) -> dict[str, Any]:
    count_payload: dict[str, Any] = {
        "contents": translated_payload.get("contents") or [],
    }
    for key in (
        "systemInstruction",
        "tools",
        "toolConfig",
        "generationConfig",
        "cachedContent",
        "safetySettings",
    ):
        value = translated_payload.get(key)
        if value is not None:
            count_payload[key] = value
    return count_payload


def _apply_max_tool_calls_limit(response_payload: dict[str, Any], max_tool_calls: Any) -> dict[str, Any]:
    if not isinstance(max_tool_calls, int) or isinstance(max_tool_calls, bool) or max_tool_calls <= 0:
        return response_payload

    output = response_payload.get("output")
    if not isinstance(output, list):
        return response_payload

    limited_output: list[dict[str, Any]] = []
    function_call_count = 0
    exceeded = False
    for item in output:
        if isinstance(item, dict) and item.get("type") == "function_call":
            function_call_count += 1
            if function_call_count > max_tool_calls:
                exceeded = True
                continue
        limited_output.append(item)

    if not exceeded:
        return response_payload

    updated = dict(response_payload)
    updated["output"] = limited_output
    updated["status"] = "incomplete"
    updated["incomplete_details"] = {"reason": "max_tool_calls"}
    updated["error"] = None
    return updated
