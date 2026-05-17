from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any

import redis


@dataclass(frozen=True)
class RedisStreamMessage:
    stream_id: str
    payload: str


class RedisStreamQueue:
    """Redis Stream queue with optional consumer-group acknowledgement."""

    def __init__(
        self,
        url: str,
        name: str,
        max_len: int = 128,
        group: str | None = None,
        consumer: str | None = None,
    ):
        self.client = redis.Redis.from_url(url)
        self.name = name
        self.max_len = max_len
        self.group = group
        self.consumer = consumer or socket.gethostname()
        if self.group:
            self._ensure_group()

    def _ensure_group(self) -> None:
        try:
            self.client.xgroup_create(self.name, self.group, id="0", mkstream=True)
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def push(self, payload: str) -> None:
        self.client.xadd(self.name, {"payload": payload}, maxlen=self.max_len, approximate=True)

    def pop(self, timeout_ms: int = 1000) -> RedisStreamMessage | None:
        if not self.group:
            raise ValueError("RedisStreamQueue.pop requires a consumer group")

        items = self.client.xreadgroup(
            self.group,
            self.consumer,
            {self.name: ">"},
            count=1,
            block=max(1, int(timeout_ms)),
        )
        if not items:
            return None
        _, messages = items[0]
        stream_id, fields = messages[0]
        payload = fields.get(b"payload")
        if payload is None:
            self.ack(stream_id.decode("utf-8"))
            return None
        return RedisStreamMessage(stream_id.decode("utf-8"), payload.decode("utf-8"))

    def ack(self, stream_id: str) -> None:
        if not self.group:
            return
        self.client.xack(self.name, self.group, stream_id)

    def peek_latest(self) -> dict[str, Any] | None:
        items = self.client.xrevrange(self.name, count=1)
        if not items:
            return None
        _, fields = items[0]
        payload = fields.get(b"payload")
        if payload is None:
            return None
        return json.loads(payload)

    def recent(self, count: int = 20) -> list[dict[str, Any]]:
        items = self.client.xrevrange(self.name, count=max(1, int(count)))
        results = []
        for stream_id, fields in items:
            payload = fields.get(b"payload")
            if payload is None:
                continue
            item = json.loads(payload)
            item["_stream_id"] = stream_id.decode("utf-8")
            results.append(item)
        return results

    def ping(self) -> bool:
        return bool(self.client.ping())
