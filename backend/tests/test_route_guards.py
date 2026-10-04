"""Admin and login guards on expensive routes, plus request bounds.

Uses TestClient without the lifespan, so no scheduler, DB or model loads.
Only requests that are refused before any real work are sent.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.auth import mint_token
from backend.config import settings

client = TestClient(main.app)

ADMIN_ROUTES = [
    ("post", "/ingestion/trigger/predictions"),
    ("post", "/predict/run"),
    ("post", "/cache/flush"),
    ("post", "/ai/insights/refresh"),
    ("get", "/predict/BTC?fresh=true"),
]

USER_ROUTES = [
    ("post", "/ai/chat", {"messages": [{"role": "user", "content": "hi"}]}),
    ("post", "/ai/chat/stream", {"messages": [{"role": "user", "content": "hi"}]}),
    ("post", "/predict/BTC/verify", None),
    ("get", "/ai/intel/BTC", None),
    ("post", "/ai/rag/query", {"query": "hi"}),
    ("post", "/backtest", {"symbol": "AAPL", "start": "2024-01-01", "end": "2024-06-01"}),
]


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_REQUIRED", True)
    monkeypatch.setattr(settings, "ADMIN_TOKEN", "")


@pytest.mark.parametrize("method,path", ADMIN_ROUTES)
def test_admin_routes_fail_closed_without_configured_token(method, path):
    r = getattr(client, method)(path, headers={"X-Admin-Token": "anything"})
    assert r.status_code == 403


@pytest.mark.parametrize("method,path", ADMIN_ROUTES)
def test_admin_routes_refuse_wrong_token(method, path, monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_TOKEN", "s3cret")
    r = getattr(client, method)(path, headers={"X-Admin-Token": "nope"})
    assert r.status_code == 403


def test_admin_token_unlocks(monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_TOKEN", "s3cret")
    r = client.post("/cache/flush", headers={"X-Admin-Token": "s3cret"})
    assert r.status_code == 200


@pytest.mark.parametrize("method,path,body", USER_ROUTES)
def test_llm_and_backtest_routes_need_login(method, path, body):
    kw = {"json": body} if body is not None else {}
    r = getattr(client, method)(path, **kw)
    assert r.status_code == 401


def _auth():
    return {"Authorization": "Bearer " + mint_token(1)}


@pytest.mark.parametrize(
    "body",
    [
        {"symbol": "AAPL; DROP", "start": "2024-01-01", "end": "2024-06-01"},
        {"symbol": "AAPL", "start": "2024-01-01", "end": "2024-06-01", "short_sma": 1},
        {"symbol": "AAPL", "start": "2024-01-01", "end": "2024-06-01", "long_sma": 5000},
        {"symbol": "AAPL", "start": "2024-01-01", "end": "2024-06-01", "initial_capital": 1e12},
        {"symbol": "AAPL", "start": "not-a-date", "end": "2024-06-01"},
    ],
)
def test_backtest_rejects_out_of_bounds(body):
    assert client.post("/backtest", json=body, headers=_auth()).status_code == 422


@pytest.mark.parametrize("start,end", [("2024-06-01", "2024-01-01"), ("2000-01-01", "2024-01-01")])
def test_backtest_rejects_bad_ranges(start, end):
    body = {"symbol": "AAPL", "start": start, "end": end}
    assert client.post("/backtest", json=body, headers=_auth()).status_code == 400


def test_chat_rejects_oversized_history():
    body = {"messages": [{"role": "user", "content": "x"}] * 51}
    assert client.post("/ai/chat", json=body, headers=_auth()).status_code == 422


def test_forecast_recompute_limited_to_model_symbols():
    syms = main._model_symbols()
    assert "BTC" in syms and "AAPL" in syms
    assert "NOPE" not in syms
