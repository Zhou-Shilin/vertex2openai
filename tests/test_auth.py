from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.auth import extract_api_key


def build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/key")
    async def read_key(request: Request):
        return {"key": extract_api_key(request)}

    return app


def test_extracts_bearer_key() -> None:
    app = build_app()
    client = TestClient(app)

    response = client.get("/key", headers={"Authorization": "Bearer bearer-key"})
    assert response.status_code == 200
    assert response.json()["key"] == "bearer-key"


def test_extracts_query_key() -> None:
    app = build_app()
    client = TestClient(app)

    response = client.get("/key?api_key=query-key")
    assert response.status_code == 200
    assert response.json()["key"] == "query-key"


def test_bearer_takes_precedence() -> None:
    app = build_app()
    client = TestClient(app)

    response = client.get(
        "/key?api_key=query-key",
        headers={"Authorization": "Bearer bearer-key"},
    )
    assert response.status_code == 200
    assert response.json()["key"] == "bearer-key"

