# Vertex2OpenAI

OpenAI Responses API-compatible proxy server for Google Vertex AI using API Key auth.

## Features

- OpenAI-style endpoints:
  - `POST /v1/responses`
  - `GET /v1/responses/{response_id}`
  - `DELETE /v1/responses/{response_id}`
  - `GET /v1/models`
- Inbound auth:
  - `Authorization: Bearer <KEY>` (preferred)
  - `?api_key=<KEY>`
- Upstream forwarding:
  - default: `https://aiplatform.googleapis.com/v1`
  - API key forwarded as `?key=<KEY>`
- Responses API support:
  - non-stream + stream SSE
  - function tool calling (`tools(type=function)`)
  - `function_call_output` continuation
  - text + image input
- Redis persistence for `store=true` responses and `previous_response_id`
- Strict OpenAI-style error object

## Quick start (local)

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Copy env file.

```bash
cp .env.example .env
```

3. Run Redis (Docker example).

```bash
docker run --rm -p 6379:6379 redis:7-alpine
```

4. Run API server.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Quick start (Docker Compose)

```bash
cp .env.example .env
docker compose up --build
```

## Example requests

Non-stream:

```bash
curl -X POST "http://localhost:8080/v1/responses" \
  -H "Authorization: Bearer YOUR_GOOGLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3-flash-preview",
    "input": "Write one sentence about FastAPI.",
    "store": true
  }'
```

Stream:

```bash
curl -N -X POST "http://localhost:8080/v1/responses" \
  -H "Authorization: Bearer YOUR_GOOGLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3-flash-preview",
    "input": "Explain SSE in 2 bullets.",
    "stream": true,
    "store": false
  }'
```

List models:

```bash
curl "http://localhost:8080/v1/models?api_key=YOUR_GOOGLE_API_KEY"
```

## Notes and compatibility

- If `GOOGLE_API_BASE` is `aiplatform.googleapis.com`, short model names are normalized to `publishers/google/models/{model}`.
- If `GOOGLE_API_BASE` is `generativelanguage.googleapis.com`, short model names are normalized to `models/{model}`.
- Only function tools are supported. Non-function tools return `400`.
- Audio input/output is not supported in this version.
- CORS is disabled by default. Configure `CORS_ALLOWED_ORIGINS` to enable.
- `store=false` means no persistence; `GET/DELETE` will return `404`.
