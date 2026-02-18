import asyncio
from typing import Any, AsyncIterator, Optional

from fastapi.testclient import TestClient

from app.config import Settings
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
            {"name": "models/gemini-2.5-flash"},
            {"name": "publishers/google/models/gemini-2.5-pro"},
        ]


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
        redis_url="redis://unused",
        response_store_ttl_seconds=86400,
    )
    app = create_app(
        settings=settings,
        injected_gemini_client=gemini,  # type: ignore[arg-type]
        injected_response_store=store,  # type: ignore[arg-type]
    )
    return TestClient(app), store, gemini


def test_create_get_delete_response_roundtrip() -> None:
    client, _, _ = build_test_client()
    with client:
        create = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key"},
            json={"model": "gemini-2.5-flash", "input": "hello", "store": True},
        )
        assert create.status_code == 200
        payload = create.json()
        response_id = payload["id"]
        assert payload["object"] == "response"

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
            json={"model": "gemini-2.5-flash", "input": "hello", "stream": True, "store": False},
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())
            assert "event: response.created" in body
            assert "event: response.output_text.delta" in body
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
            "model": "gemini-2.5-flash",
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
        "model": "gemini-2.5-flash",
        "status": "completed",
    }
    store.data["resp_prev"] = previous

    with client:
        response = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key"},
            json={
                "model": "gemini-2.5-flash",
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
        assert payload["data"][1]["id"] == "gemini-2.5-pro"
