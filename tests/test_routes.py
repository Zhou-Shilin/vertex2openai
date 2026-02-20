import asyncio
from typing import Any, AsyncIterator, Optional

from fastapi.testclient import TestClient

from app.config import Settings
from app.errors import APIError
from app.main import create_app


class InMemoryStore:
    def __init__(self) -> None:
        self.data: dict[str, dict[str, Any]] = {}

    async def save(self, record: dict[str, Any]) -> None:
        self.data[record["id"]] = record

    async def get(self, response_id: str) -> Optional[dict[str, Any]]:
        return self.data.get(response_id)

    async def delete(self, response_id: str) -> bool:
        return self.data.pop(response_id, None) is not None


class FakeGeminiClient:
    def __init__(self) -> None:
        self.last_payload: Optional[dict[str, Any]] = None

    async def generate_content(self, model: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        self.last_payload = payload
        if payload.get("tools"):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "lookup_weather",
                                        "args": {"city": "Paris"},
                                    }
                                }
                            ]
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
            }
        return {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
        }

    async def stream_generate_content(
        self,
        model: str,
        payload: dict[str, Any],
        api_key: str,
    ) -> AsyncIterator[dict[str, Any]]:
        self.last_payload = payload
        yield {
            "candidates": [{"content": {"parts": [{"text": "Hel"}]}}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
        }
        await asyncio.sleep(0)
        yield {
            "candidates": [{"content": {"parts": [{"text": "Hello"}]}}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2, "totalTokenCount": 3},
        }

    async def list_models(self, api_key: str) -> list[dict[str, Any]]:
        return [
            {"name": "models/gemini-3-flash-preview"},
            {"name": "publishers/google/models/gemini-2.5-flash"},
        ]

    async def count_tokens(self, model: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        self.last_payload = payload
        return {"totalTokens": 7}


class FailingListGeminiClient(FakeGeminiClient):
    async def list_models(self, api_key: str) -> list[dict[str, Any]]:
        raise APIError(
            "not found",
            404,
            error_type="not_found_error",
            code="not_found",
        )


class SlowGeminiClient(FakeGeminiClient):
    async def generate_content(self, model: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        await asyncio.sleep(0.2)
        return await super().generate_content(model, payload, api_key)


class MultiFunctionGeminiClient(FakeGeminiClient):
    async def generate_content(self, model: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        self.last_payload = payload
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "tool_one", "args": {"x": 1}}},
                            {"functionCall": {"name": "tool_two", "args": {"y": 2}}},
                        ]
                    }
                }
            ],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2, "totalTokenCount": 3},
        }


def build_test_client() -> tuple[TestClient, InMemoryStore, FakeGeminiClient]:
    store = InMemoryStore()
    gemini = FakeGeminiClient()
    settings = Settings(
        host="127.0.0.1",
        port=8080,
        log_level="INFO",
        log_redact_body=True,
        request_body_log_max_chars=512,
        cors_allowed_origins=(),
        google_api_base="https://generativelanguage.googleapis.com/v1beta",
        upstream_first_byte_timeout_seconds=120,
        upstream_stream_read_timeout_seconds=0,
        fallback_models=("gemini-3-flash-preview", "gemini-2.5-flash"),
        redis_url="redis://unused",
        response_store_ttl_seconds=86400,
    )
    app = create_app(
        settings=settings,
        injected_gemini_client=gemini,  # type: ignore[arg-type]
        injected_response_store=store,  # type: ignore[arg-type]
    )
    return TestClient(app), store, gemini


def build_test_client_with_failing_models() -> TestClient:
    store = InMemoryStore()
    gemini = FailingListGeminiClient()
    settings = Settings(
        host="127.0.0.1",
        port=8080,
        log_level="INFO",
        log_redact_body=True,
        request_body_log_max_chars=512,
        cors_allowed_origins=(),
        google_api_base="https://generativelanguage.googleapis.com/v1beta",
        upstream_first_byte_timeout_seconds=120,
        upstream_stream_read_timeout_seconds=0,
        fallback_models=("gemini-3-flash-preview", "gemini-2.5-flash"),
        redis_url="redis://unused",
        response_store_ttl_seconds=86400,
    )
    app = create_app(
        settings=settings,
        injected_gemini_client=gemini,  # type: ignore[arg-type]
        injected_response_store=store,  # type: ignore[arg-type]
    )
    return TestClient(app)


def test_create_get_delete_response_roundtrip() -> None:
    client, _, _ = build_test_client()
    with client:
        create = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key"},
            json={
                "model": "gemini-3-flash-preview",
                "input": "hello",
                "store": True,
                "metadata": {"trace_id": "abc"},
                "user": "user-1",
            },
        )
        assert create.status_code == 200
        payload = create.json()
        response_id = payload["id"]
        assert payload["object"] == "response"
        assert payload["metadata"]["trace_id"] == "abc"
        assert payload["user"] == "user-1"

        read = client.get(f"/v1/responses/{response_id}", headers={"Authorization": "Bearer key"})
        assert read.status_code == 200
        assert read.json()["id"] == response_id

        deleted = client.delete(f"/v1/responses/{response_id}", headers={"Authorization": "Bearer key"})
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True

        missing = client.get(f"/v1/responses/{response_id}", headers={"Authorization": "Bearer key"})
        assert missing.status_code == 404
        assert "error" in missing.json()


def test_stream_response_events() -> None:
    client, _, _ = build_test_client()
    with client:
        with client.stream(
            "POST",
            "/v1/responses",
            headers={"Authorization": "Bearer key"},
            json={"model": "gemini-3-flash-preview", "input": "hello", "stream": True, "store": False},
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())
            assert "event: response.created" in body
            assert "event: response.content_part.added" in body
            assert "event: response.output_text.delta" in body
            assert "event: response.output_text.done" in body
            assert "event: response.output_item.done" in body
            assert "event: response.completed" in body
            assert "data: [DONE]" in body


def test_previous_response_id_tool_output_flow() -> None:
    client, store, gemini = build_test_client()
    previous = {
            "id": "resp_prev",
            "request": {},
            "response": {
                "id": "resp_prev",
                "object": "response",
                "model": "gemini-3-flash-preview",
            "output": [
                {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "lookup_weather",
                    "arguments": '{"city":"Paris"}',
                    "status": "completed",
                }
            ],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "created_at": 0,
        },
        "created_at": 0,
        "expires_at": 9999999999,
        "model": "gemini-3-flash-preview",
        "status": "completed",
    }
    store.data["resp_prev"] = previous

    with client:
        response = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key"},
            json={
                "model": "gemini-3-flash-preview",
                "previous_response_id": "resp_prev",
                "input": [{"type": "function_call_output", "call_id": "call_abc", "output": '{"ok":true}'}],
                "store": False,
            },
        )
        assert response.status_code == 200
        assert gemini.last_payload is not None
        last_content = gemini.last_payload["contents"][-1]
        function_response = last_content["parts"][0]["functionResponse"]
        assert function_response["name"] == "lookup_weather"
        assert function_response["response"]["ok"] is True


def test_models_endpoint() -> None:
    client, _, _ = build_test_client()
    with client:
        response = client.get("/v1/models?api_key=query-key")
        assert response.status_code == 200
        payload = response.json()
        assert payload["object"] == "list"
        assert payload["data"][0]["id"] == "gemini-2.5-flash"
        assert payload["data"][1]["id"] == "gemini-3-flash-preview"


def test_models_endpoint_fallback_when_upstream_list_unavailable() -> None:
    client = build_test_client_with_failing_models()
    with client:
        response = client.get("/v1/models?api_key=query-key")
        assert response.status_code == 200
        payload = response.json()
        assert payload["object"] == "list"
        assert payload["data"][0]["id"] == "gemini-2.5-flash"
        assert payload["data"][1]["id"] == "gemini-3-flash-preview"


def test_response_input_items_endpoint() -> None:
    client, _, _ = build_test_client()
    with client:
        create = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key"},
            json={
                "model": "gemini-3-flash-preview",
                "store": True,
                "instructions": "be concise",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hello"}],
                    }
                ],
            },
        )
        assert create.status_code == 200
        response_id = create.json()["id"]

        listed = client.get(f"/v1/responses/{response_id}/input_items", headers={"Authorization": "Bearer key"})
        assert listed.status_code == 200
        payload = listed.json()
        assert payload["object"] == "list"
        assert payload["has_more"] is False
        assert len(payload["data"]) == 2
        assert payload["data"][0]["role"] == "system"
        assert payload["data"][1]["role"] == "user"


def test_response_input_tokens_endpoint() -> None:
    client, _, gemini = build_test_client()
    with client:
        response = client.post(
            "/v1/responses/input_tokens",
            headers={"Authorization": "Bearer key"},
            json={
                "model": "gemini-3-flash-preview",
                "input": "hello",
                "instructions": "be concise",
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["object"] == "response.input_tokens"
        assert payload["model"] == "gemini-3-flash-preview"
        assert payload["input_tokens"] == 7
        assert gemini.last_payload is not None
        assert isinstance(gemini.last_payload.get("contents"), list)


def test_response_compact_endpoint_returns_stored_response() -> None:
    client, _, _ = build_test_client()
    with client:
        create = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key"},
            json={
                "model": "gemini-3-flash-preview",
                "input": "hello",
                "store": True,
            },
        )
        assert create.status_code == 200
        response_id = create.json()["id"]

        compact = client.post(f"/v1/responses/{response_id}/compact", headers={"Authorization": "Bearer key"})
        assert compact.status_code == 200
        assert compact.json()["id"] == response_id


def test_max_tool_calls_limits_function_call_output() -> None:
    store = InMemoryStore()
    gemini = MultiFunctionGeminiClient()
    settings = Settings(
        host="127.0.0.1",
        port=8080,
        log_level="INFO",
        log_redact_body=True,
        request_body_log_max_chars=512,
        cors_allowed_origins=(),
        google_api_base="https://generativelanguage.googleapis.com/v1beta",
        upstream_first_byte_timeout_seconds=120,
        upstream_stream_read_timeout_seconds=0,
        fallback_models=("gemini-3-flash-preview", "gemini-2.5-flash"),
        redis_url="redis://unused",
        response_store_ttl_seconds=86400,
    )
    app = create_app(
        settings=settings,
        injected_gemini_client=gemini,  # type: ignore[arg-type]
        injected_response_store=store,  # type: ignore[arg-type]
    )
    client = TestClient(app)

    with client:
        response = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key"},
            json={
                "model": "gemini-3-flash-preview",
                "input": "hello",
                "tools": [
                    {"type": "function", "name": "tool_one", "parameters": {"type": "object"}},
                    {"type": "function", "name": "tool_two", "parameters": {"type": "object"}},
                ],
                "max_tool_calls": 1,
                "store": False,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        function_calls = [item for item in payload["output"] if item.get("type") == "function_call"]
        assert len(function_calls) == 1
        assert payload["status"] == "incomplete"
        assert payload["incomplete_details"] == {"reason": "max_tool_calls"}


def test_background_response_and_cancel() -> None:
    store = InMemoryStore()
    gemini = SlowGeminiClient()
    settings = Settings(
        host="127.0.0.1",
        port=8080,
        log_level="INFO",
        log_redact_body=True,
        request_body_log_max_chars=512,
        cors_allowed_origins=(),
        google_api_base="https://generativelanguage.googleapis.com/v1beta",
        upstream_first_byte_timeout_seconds=120,
        upstream_stream_read_timeout_seconds=0,
        fallback_models=("gemini-3-flash-preview", "gemini-2.5-flash"),
        redis_url="redis://unused",
        response_store_ttl_seconds=86400,
    )
    app = create_app(
        settings=settings,
        injected_gemini_client=gemini,  # type: ignore[arg-type]
        injected_response_store=store,  # type: ignore[arg-type]
    )
    client = TestClient(app)

    with client:
        created = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key"},
            json={
                "model": "gemini-3-flash-preview",
                "input": "hello",
                "background": True,
                "store": True,
            },
        )
        assert created.status_code == 200
        payload = created.json()
        response_id = payload["id"]
        assert payload["status"] in {"queued", "in_progress", "completed"}

        cancelled = client.post(f"/v1/responses/{response_id}/cancel", headers={"Authorization": "Bearer key"})
        assert cancelled.status_code == 200
        cancelled_payload = cancelled.json()
        assert cancelled_payload["id"] == response_id
        assert cancelled_payload["status"] in {"incomplete", "completed", "failed"}
