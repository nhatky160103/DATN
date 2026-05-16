from __future__ import annotations

from typing import Any

import redis


class RedisListQueue:
    """Small Redis-backed FIFO queue using LPUSH/BRPOP."""

    def __init__(self, url: str, name: str, max_size: int = 128):
        self.client = redis.Redis.from_url(url)
        self.name = name
        self.max_size = max_size

    def push(self, payload: str) -> None:
        pipe = self.client.pipeline()
        pipe.lpush(self.name, payload)
        pipe.ltrim(self.name, 0, self.max_size - 1)
        pipe.execute()

    def pop(self, timeout_sec: int = 1) -> str | None:
        item = self.client.brpop(self.name, timeout=timeout_sec)
        if item is None:
            return None
        _, payload = item
        return payload.decode("utf-8")

    def peek_latest(self) -> dict[str, Any] | None:
        raw = self.client.lindex(self.name, 0)
        if raw is None:
            return None
        import json

        return json.loads(raw)

    def ping(self) -> bool:
        return bool(self.client.ping())

