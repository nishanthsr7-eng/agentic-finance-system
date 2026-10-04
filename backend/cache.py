"""
FLUX Backend — In-Memory TTL Cache
Zero-dependency dict cache. Thread-safe for single-process uvicorn.
Upgrade to Redis when horizontal scaling is needed.
"""

import time
from typing import Any


class Cache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Any | None:
        """Return cached value or None if missing / expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: int = 60) -> None:
        """Store value with a TTL in seconds."""
        self._store[key] = (value, time.monotonic() + ttl)

    def invalidate(self, key: str) -> None:
        """Force-expire a key."""
        self._store.pop(key, None)

    def stats(self) -> dict:
        """Return live/expired counts — useful for the /health endpoint."""
        now = time.monotonic()
        live = sum(1 for _, exp in self._store.values() if exp > now)
        return {"total": len(self._store), "live": live, "expired": len(self._store) - live}


# Single shared instance — imported by main.py
cache = Cache()
