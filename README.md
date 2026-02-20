# Vertex2OpenAI

OpenAI Responses API-compatible proxy server for Google Vertex AI using API Key auth.

## Features

- OpenAI-style endpoints:
  - `POST /v1/responses`
  - `POST /v1/responses/input_tokens`
  - `GET /v1/responses/{response_id}`
  - `GET /v1/responses/{response_id}/input_items`
  - `POST /v1/responses/{response_id}/cancel`
  - `POST /v1/responses/{response_id}/compact`
  - `DELETE /v1/responses/{response_id}`
  - `GET /v1/models`
- Inbound auth:
  - `Authorization: Bearer <KEY>` (preferred)
  - `?api_key=<KEY>`
- Upstream forwarding:
  - default: `https://aiplatform.googleapis.com/v1`
  - API key forwarded as `?key=<KEY>`
- Model list fallback:
  - when upstream model listing endpoint is unavailable, `/v1/models` returns `FALLBACK_MODELS`
  - default includes `gemini-3-flash-preview`
- Responses API support:
  - non-stream + stream SSE + background mode (`background=true`)
  - tool calling:
    - `tools(type=function)` -> Vertex function declarations
    - `tools(type=web_search_preview*)` -> Vertex `googleSearch`
    - `tools(type=code_interpreter)` -> Vertex `codeExecution`
    - `tools(type=file_search|computer_use_preview|image_generation|mcp|local_shell|custom|shell|apply_patch)` -> synthetic function declarations
    - `tool_choice(type=allowed_tools)` -> Vertex `allowedFunctionNames`
  - `function_call_output` continuation
  - text + image + file input (`input_file.file_data`) + audio input (`input_audio`)
  - `top_logprobs` -> Vertex `responseLogprobs/logprobs`
  - `modalities` + `audio.voice` -> Vertex `responseModalities/speechConfig`
  - `max_tool_calls` local enforcement for function-call outputs
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
- For OpenAI `file_id` image/file references, provide `file_id_map` (or `vertex.file_id_map`) to resolve IDs to URI/data URL.
- Audio input is supported (`input_audio`). Audio output supports `modalities` and `audio.voice`; precise output encoding can be set with `vertex.generation_config`.
- CORS is disabled by default. Configure `CORS_ALLOWED_ORIGINS` to enable.
- Configure `FALLBACK_MODELS` if your key cannot call upstream `models.list`.
- `store=false` means no persistence; `GET/DELETE` will return `404`.
