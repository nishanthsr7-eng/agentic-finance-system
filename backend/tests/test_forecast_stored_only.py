"""The forecast route serves stored predictions and never runs the model.

Running inference inside a page request pushed the 512 MB host over its limit,
so only the daily job and the admin trigger may call the model.
Uses TestClient without the lifespan, so no scheduler, DB or model loads.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import db, main
from backend.prediction import serve
from backend.routes import predict

client = TestClient(main.app)

HISTORY = [
    {
        "date": f"2026-09-{d:02d}",
        "open": 1.0,
        "high": 2.0,
        "low": 0.5,
        "close": 1.5,
        "adj_close": 1.5,
    }
    for d in range(1, 21)
]


@pytest.fixture(autouse=True)
def _no_model(monkeypatch):
    async def history(symbol, start=None):
        return HISTORY

    async def boom(*a, **k):
        raise AssertionError("the forecast route must not run the model")

    monkeypatch.setattr(db, "get_history", history)
    monkeypatch.setattr(serve, "predict_and_log", boom)


def _stored(rows):
    async def latest(symbol, limit):
        return rows

    return latest


def test_old_stored_prediction_is_served_as_is(monkeypatch):
    old = {"symbol": "BTC", "direction": "UP", "generated_at": 1_700_000_000_000}
    monkeypatch.setattr(predict, "get_latest_predictions", _stored([old]))
    r = client.get("/predict/BTC/forecast")
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == old
    assert body["forecastable"] is True
    assert body["count"] == len(HISTORY)


def test_missing_prediction_returns_none_without_inference(monkeypatch):
    monkeypatch.setattr(predict, "get_latest_predictions", _stored([]))
    body = client.get("/predict/BTC/forecast").json()
    assert body["prediction"] is None
    assert body["forecastable"] is True


def test_symbol_outside_the_model_is_flagged(monkeypatch):
    monkeypatch.setattr(predict, "get_latest_predictions", _stored([]))
    body = client.get("/predict/USDT/forecast").json()
    assert body["prediction"] is None
    assert body["forecastable"] is False
