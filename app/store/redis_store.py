"""Redis persistence for stored OpenAI responses."""

from __future__ import annotations

import json
from typing import Any, Optional

import redis.asyncio as redis


class RedisResponseStore:
    def __init__(self, redis_url: str, ttl_seconds: int, *, key_prefix: str = "vertex2openai:resp") -> None:
        self._redis_url = redis_url
        self._ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix
        self._client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        self._client = redis.from_url(self._redis_url, encoding="utf-8", decode_responses=True)
        await self._client.ping()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _require_client(self) -> redis.Redis:
        if self._client is None:
            raise RuntimeError("Redis client is not connected.")
        return self._client

    def _key(self, response_id: str) -> str:
        return f"{self._key_prefix}:{response_id}"

    async def save(self, record: dict[str, Any]) -> None:
        client = self._require_client()
        await client.set(self._key(record["id"]), json.dumps(record, separators=(",", ":")), ex=self._ttl_seconds)

    async def get(self, response_id: str) -> Optional[dict[str, Any]]:
        client = self._require_client()
        raw = await client.get(self._key(response_id))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def delete(self, response_id: str) -> bool:
        client = self._require_client()
        deleted = await client.delete(self._key(response_id))
        return deleted > 0
