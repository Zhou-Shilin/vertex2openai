"""Translate Gemini REST payload into OpenAI Responses format."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


def new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def new_item_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex}"


def usage_from_google(usage: Optional[dict[str, Any]]) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = int(usage.get("promptTokenCount") or 0)
    output_tokens = int(usage.get("candidatesTokenCount") or 0)
    total_tokens = int(usage.get("totalTokenCount") or (input_tokens + output_tokens))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def make_message_output_item(text: str, *, item_id: Optional[str] = None) -> dict[str, Any]:
    return {
        "id": item_id or new_item_id("msg"),
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def make_function_call_item(
    name: str,
    arguments: str,
    *,
    call_id: Optional[str] = None,
    item_id: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "id": item_id or new_item_id("fc"),
        "type": "function_call",
        "status": "completed",
        "call_id": call_id or new_call_id(),
        "name": name,
        "arguments": arguments,
    }


def make_response_object(
    *,
    response_id: str,
    model: str,
    output: list[dict[str, Any]],
    usage: Optional[dict[str, int]] = None,
    status: str = "completed",
    created_at: Optional[int] = None,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at or int(time.time()),
        "status": status,
        "error": None,
        "incomplete_details": None,
        "model": model,
        "output": output,
        "usage": usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


def translate_google_response_to_openai(
    *,
    google_payload: dict[str, Any],
    model: str,
    response_id: str,
) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    text_parts: list[str] = []

    for candidate in google_payload.get("candidates", []) or []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else []
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                text_parts.append(text)
            function_call = part.get("functionCall")
            if isinstance(function_call, dict):
                name = function_call.get("name")
                if not isinstance(name, str) or not name:
                    continue
                args = function_call.get("args")
                arguments = json.dumps(args if args is not None else {}, ensure_ascii=False, separators=(",", ":"))
                output.append(make_function_call_item(name, arguments))

    if text_parts:
        output.insert(0, make_message_output_item("".join(text_parts)))

    return make_response_object(
        response_id=response_id,
        model=model,
        output=output,
        usage=usage_from_google(google_payload.get("usageMetadata")),
        status="completed",
    )


def build_stored_record(
    *,
    response_id: str,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
    model: str,
    ttl_seconds: int,
) -> dict[str, Any]:
    created_at = int(time.time())
    return {
        "id": response_id,
        "request": request_payload,
        "response": response_payload,
        "created_at": created_at,
        "expires_at": created_at + ttl_seconds,
        "model": model,
        "status": response_payload.get("status", "completed"),
    }


def sse_event(event_type: str, payload: dict[str, Any]) -> str:
    body = {"type": event_type}
    body.update(payload)
    return f"event: {event_type}\ndata: {json.dumps(body, ensure_ascii=False, separators=(',', ':'))}\n\n"


@dataclass
class StreamAggregateState:
    response_id: str
    model: str
    created_at: int = field(default_factory=lambda: int(time.time()))
    text_item_id: str = field(default_factory=lambda: new_item_id("msg"))
    text_value: str = ""
    text_item_emitted: bool = False
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    function_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    output_order: list[str] = field(default_factory=list)

    def update_usage(self, usage_metadata: Optional[dict[str, Any]]) -> None:
        usage = usage_from_google(usage_metadata)
        if usage["total_tokens"] > 0:
            self.usage = usage

    def register_function_call(self, *, key: str, name: str, arguments: str) -> tuple[dict[str, Any], bool]:
        existing = self.function_calls.get(key)
        if existing is None:
            item = make_function_call_item(name, arguments)
            self.function_calls[key] = item
            self.output_order.append(item["id"])
            return item, True
        existing["arguments"] = arguments
        return existing, False

    def set_text(self, incoming_text: str) -> str:
        if not incoming_text:
            return ""
        if incoming_text.startswith(self.text_value):
            delta = incoming_text[len(self.text_value) :]
            self.text_value = incoming_text
            return delta
        if self.text_value.startswith(incoming_text):
            return ""
        self.text_value += incoming_text
        return incoming_text

    def build_final_response(self) -> dict[str, Any]:
        output: list[dict[str, Any]] = []
        if self.text_value:
            output.append(make_message_output_item(self.text_value, item_id=self.text_item_id))
        output.extend(self.function_calls.values())
        return make_response_object(
            response_id=self.response_id,
            model=self.model,
            output=output,
            usage=self.usage,
            status="completed",
            created_at=self.created_at,
        )
