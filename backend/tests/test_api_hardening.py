"""Chat roles are restricted, errors don't echo internals, CORS lists its headers,
/db/health only shows row counts to the admin, and /health and /ingestion/status
keep internals behind the admin token.

Uses TestClient without the lifespan, so no scheduler, DB or model loads.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend import llm, main, user_api
from backend.auth import require_user
from backend.routes import backtest

client = TestClient(main.app)
ORIGIN = "http://localhost:3000"


def _as_user():
    main.app.dependency_overrides[require_user] = lambda: 1


def teardown_function():
    main.app.dependency_overrides.clear()


def test_chat_rejects_unknown_role():
    _as_user()
    r = client.post("/ai/chat", json={"messages": [{"role": "tool", "content": "hi"}]})
    assert r.status_code == 422


def test_chat_error_hides_provider_text(monkeypatch):
    _as_user()

    async def boom(*_a, **_k):
        raise llm.LLMError("LLM HTTP 401: invalid key gsk_secret")

    monkeypatch.setattr(llm, "chat", boom)
    r = client.post("/ai/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503
    assert "gsk_secret" not in r.text and "401" not in r.text


def test_backtest_error_hides_exception_text(monkeypatch):
    _as_user()

    def boom(*_a):
        raise RuntimeError("internal path C:/secret/frame.py")

    monkeypatch.setattr(backtest, "_run_backtest_sync", boom)
    body = {"symbol": "BTC-USD", "start": "2024-01-01", "end": "2024-06-01"}
    r = client.post("/backtest", json=body)
    assert r.status_code == 502
    assert "secret" not in r.text


def test_backtest_data_error_is_shown(monkeypatch):
    _as_user()

    def short(*_a):
        raise backtest.BacktestDataError("Only 10 days of BTC-USD data; SMA(50) needs more")

    monkeypatch.setattr(backtest, "_run_backtest_sync", short)
    body = {"symbol": "BTC-USD", "start": "2024-01-01", "end": "2024-01-15"}
    r = client.post("/backtest", json=body)
    assert r.status_code == 404
    assert "Only 10 days" in r.json()["detail"]


def _preflight(headers: str):
    return client.options(
        "/ai/chat",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": headers,
        },
    )


def test_cors_allows_frontend_headers():
    assert _preflight("authorization,content-type").status_code == 200


def test_cors_refuses_other_headers():
    assert _preflight("x-evil").status_code == 400


def test_db_health_public_is_trimmed(monkeypatch):
    monkeypatch.setattr(user_api.M, "ping", lambda: True)
    assert client.get("/db/health").json() == {"ok": True}


def test_db_health_admin_gets_counts(monkeypatch):
    monkeypatch.setattr(user_api.M, "ping", lambda: True)
    monkeypatch.setattr(user_api.M, "query_one", lambda _sql: {"n": 3})
    monkeypatch.setattr(user_api.M.settings, "ADMIN_TOKEN", "t0k")
    monkeypatch.setattr(user_api, "is_admin", lambda tok: tok == "t0k")
    r = client.get("/db/health", headers={"X-Admin-Token": "t0k"})
    assert r.json()["counts"]["users"] == 3


def test_health_public_is_minimal():
    assert client.get("/health").json() == {"status": "ok"}
    assert client.head("/health").status_code == 200


def test_ingestion_status_needs_admin():
    assert client.get("/ingestion/status").status_code in (401, 403)
