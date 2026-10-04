"""Per-IP rate limits on login, register and the LLM routes."""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from backend import main
from backend.auth import mint_token
from backend.config import settings
from backend.ratelimit import rate_limit

app = FastAPI()


@app.get("/x", dependencies=[Depends(rate_limit("x", 3, 60))])
def x():
    return {"ok": True}


@app.get("/y", dependencies=[Depends(rate_limit("x", 3, 60))])
def y():
    return {"ok": True}


client = TestClient(app)


def test_limit_then_429_with_retry_after():
    assert [client.get("/x").status_code for _ in range(3)] == [200, 200, 200]
    r = client.get("/x")
    assert r.status_code == 429
    assert 1 <= int(r.headers["Retry-After"]) <= 60


def test_bucket_is_shared_across_routes():
    for _ in range(3):
        client.get("/x")
    assert client.get("/y").status_code == 429


def test_limits_are_per_client_ip():
    for _ in range(3):
        client.get("/x", headers={"CF-Connecting-IP": "1.1.1.1"})
    assert client.get("/x", headers={"CF-Connecting-IP": "1.1.1.1"}).status_code == 429
    assert client.get("/x", headers={"CF-Connecting-IP": "2.2.2.2"}).status_code == 200


def test_forwarded_for_uses_first_hop():
    for _ in range(3):
        client.get("/x", headers={"X-Forwarded-For": "9.9.9.9, 10.0.0.1"})
    assert client.get("/x", headers={"X-Forwarded-For": "9.9.9.9, 10.0.0.2"}).status_code == 429


def test_window_expires(monkeypatch):
    import backend.ratelimit as R

    t = [1000.0]
    monkeypatch.setattr(R.time, "monotonic", lambda: t[0])
    for _ in range(3):
        client.get("/x")
    assert client.get("/x").status_code == 429
    t[0] += 61
    assert client.get("/x").status_code == 200


# ── wired onto the real app ─────────────────────────────────────────────────

real = TestClient(main.app)


def test_register_limited_to_3_per_minute():
    # Invalid bodies are refused with 422 before the handler touches the DB,
    # but the limiter has already counted them.
    codes = [real.post("/auth/register", json={}).status_code for _ in range(4)]
    assert codes[:3] == [422, 422, 422] and codes[3] == 429


def test_login_limited_to_10_per_minute():
    codes = [real.post("/auth/login", json={}).status_code for _ in range(11)]
    assert codes[-1] == 429 and 429 not in codes[:10]


def test_ai_routes_share_20_per_minute(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_REQUIRED", True)
    h = {"Authorization": "Bearer " + mint_token(1)}
    bad = {"messages": []}  # 422 after the limiter counts it
    codes = [real.post("/ai/chat", json=bad, headers=h).status_code for _ in range(20)]
    assert 429 not in codes
    assert real.post("/ai/chat/stream", json=bad, headers=h).status_code == 429


@pytest.mark.parametrize("path", ["/health"])
def test_health_is_never_limited(path):
    assert all(real.head(path).status_code != 429 for _ in range(50))
