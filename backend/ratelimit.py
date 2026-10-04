"""
In-memory, per-IP sliding-window rate limits.

One process on one Render instance, so a dict in memory is enough: no Redis,
no extra dependency. Limits reset on restart, which is fine for spam control.

    @router.post("/login", dependencies=[Depends(rate_limit("login", 10, 60))])

Buckets are shared by name, so every route using "ai" draws on one allowance.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque

from fastapi import HTTPException, Request

_MAX_KEYS = 10_000  # hard cap on tracked (bucket, ip) pairs
_hits: dict[tuple[str, str], deque[float]] = {}
_lock = threading.Lock()  # sync routes run in a thread pool


def client_ip(request: Request) -> str:
    """Best guess at the caller's address behind Cloudflare and Render's proxy.

    Proxy-set headers first (a client can't forge them through the proxy),
    then the first X-Forwarded-For hop, then the socket peer."""
    for h in ("cf-connecting-ip", "true-client-ip"):
        v = request.headers.get(h)
        if v:
            return v.strip()
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _prune(now: float, window: float) -> None:
    for key in [k for k, q in _hits.items() if not q or q[-1] <= now - window]:
        del _hits[key]


def rate_limit(bucket: str, limit: int, window: float = 60.0):
    """FastAPI dependency allowing `limit` calls per `window` seconds per IP."""

    def _check(request: Request) -> None:
        now = time.monotonic()
        key = (bucket, client_ip(request))
        with _lock:
            if key not in _hits and len(_hits) >= _MAX_KEYS:
                _prune(now, window)
            q = _hits.setdefault(key, deque())
            while q and q[0] <= now - window:
                q.popleft()
            if len(q) >= limit:
                retry = max(1, math.ceil(q[0] + window - now))
                raise HTTPException(
                    status_code=429,
                    detail=f"Too many requests. Try again in {retry} s.",
                    headers={"Retry-After": str(retry)},
                )
            q.append(now)

    return _check


def reset() -> None:
    """Forget all hits (tests)."""
    with _lock:
        _hits.clear()
