"""Namespaced Redis progress events and cache operations."""

from __future__ import annotations

import json
from typing import Any


class RedisBus:
    def __init__(self, url: str):
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("Install persistence support with: pip install '.[storage]'") from exc
        self.client = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=3)

    def ping(self) -> None:
        self.client.ping()

    def publish(self, run_id: str, event: dict[str, Any]) -> None:
        payload = json.dumps(event, sort_keys=True, default=str)
        self.client.publish(f"luna:run:{run_id}:events", payload)

    def clear_run_cache(self, run_id: str) -> int:
        keys = list(self.client.scan_iter(match=f"luna:run:{run_id}:cache:*", count=100))
        return self.client.delete(*keys) if keys else 0


class NullBus:
    def __init__(self): self.events: list[dict[str, Any]] = []
    def ping(self) -> None: pass
    def publish(self, run_id: str, event: dict[str, Any]) -> None: self.events.append(event)
    def clear_run_cache(self, run_id: str) -> int: return 0

