"""Paper-trade execution: server-side price and the locked wallet update.

No database or network: the MySQL connection and the quote pools are faked.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from backend import trading_api as T
from backend.cache import cache


class FakeCursor:
    def __init__(self, cash: float, trades: list[dict]):
        self.cash = cash
        self.trades = trades
        self.sql: list[str] = []
        self.lastrowid = 42
        self._rows: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.sql.append(sql)
        if "FROM trading_wallet" in sql:
            self._rows = [{"cash_usd": self.cash}]
        elif "FROM trades" in sql:
            self._rows = list(self.trades)
        elif sql.startswith("UPDATE trading_wallet"):
            self.cash = params[0]
        elif sql.startswith("INSERT INTO trades"):
            self.inserted = params

    def fetchone(self):
        return self._rows[0]

    def fetchall(self):
        return self._rows


class FakeConn:
    def __init__(self, cur: FakeCursor):
        self.cur = cur
        self.events: list[str] = []

    def cursor(self):
        return self.cur

    def begin(self):
        self.events.append("begin")

    def commit(self):
        self.events.append("commit")

    def rollback(self):
        self.events.append("rollback")


@pytest.fixture
def market(monkeypatch):
    """BTC at $50,000 in the crypto pool; no upstream refresh."""
    cache.set("crypto", [{"symbol": "BINANCE:BTCUSDT", "sub": "BTC", "price": 50_000}], ttl=600)
    monkeypatch.setattr(T, "_refresh_quote_pool", lambda asset_type: None)
    monkeypatch.setattr(T, "_ensure_wallet", lambda uid: {"cash_usd": 0, "seeded_usd": 0})
    monkeypatch.setattr(T, "_wallet_payload", lambda uid: {"positions": []})
    yield
    cache.invalidate("crypto")


def _wire(monkeypatch, cash=100_000.0, trades=()):
    cur = FakeCursor(cash, list(trades))
    conn = FakeConn(cur)

    class _Ctx:
        def __enter__(self):
            return conn

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(T.M, "get_conn", lambda *a, **k: _Ctx())
    return conn, cur


def _order(**kw):
    base = {"symbol": "BTC", "side": "BUY", "price": 50_000, "quantity": 1, "asset_type": "crypto"}
    base.update(kw)
    return T.TradeRequest(**base)


def test_buy_fills_at_server_price_not_client_price(market, monkeypatch):
    conn, cur = _wire(monkeypatch)
    # Client offers to pay more than market: fill happens at the live quote.
    out = T.create_trade(_order(price=51_000), user_id=1)
    assert out["trade"]["price"] == 50_000
    assert cur.cash == pytest.approx(50_000)
    assert conn.events == ["begin", "commit"]
    assert any("FOR UPDATE" in q for q in cur.sql)


def test_buy_far_below_market_is_refused(market, monkeypatch):
    conn, cur = _wire(monkeypatch)
    with pytest.raises(HTTPException) as e:
        T.create_trade(_order(price=0.01), user_id=1)
    assert e.value.status_code == 409
    assert conn.events == []  # refused before touching the wallet


def test_sell_far_above_market_is_refused(market, monkeypatch):
    _wire(monkeypatch, trades=[{"symbol": "BTC", "side": "BUY", "quantity": 2}])
    with pytest.raises(HTTPException) as e:
        T.create_trade(_order(side="SELL", price=1_000_000), user_id=1)
    assert e.value.status_code == 409


def test_buy_within_tolerance_fills(market, monkeypatch):
    _wire(monkeypatch)
    out = T.create_trade(_order(price=50_000 * 0.995), user_id=1)
    assert out["trade"]["price"] == 50_000


def test_amount_order_sizes_from_server_price(market, monkeypatch):
    _wire(monkeypatch)
    out = T.create_trade(_order(quantity=None, amount=1_000, price=49_900), user_id=1)
    assert out["trade"]["quantity"] == pytest.approx(0.02)


def test_insufficient_cash_rolls_back(market, monkeypatch):
    conn, cur = _wire(monkeypatch, cash=10.0)
    with pytest.raises(HTTPException) as e:
        T.create_trade(_order(), user_id=1)
    assert e.value.status_code == 400
    assert conn.events == ["begin", "rollback"]
    assert cur.cash == 10.0


def test_sell_more_than_held_rolls_back(market, monkeypatch):
    conn, _ = _wire(
        monkeypatch,
        trades=[
            {"symbol": "BTC", "side": "BUY", "quantity": 1},
            {"symbol": "btc", "side": "SELL", "quantity": 0.5},
        ],
    )
    with pytest.raises(HTTPException) as e:
        T.create_trade(_order(side="SELL", quantity=0.6), user_id=1)
    assert e.value.status_code == 400
    assert conn.events == ["begin", "rollback"]


def test_sell_adds_cash(market, monkeypatch):
    _, cur = _wire(monkeypatch, cash=0.0, trades=[{"symbol": "BTC", "side": "BUY", "quantity": 1}])
    T.create_trade(_order(side="SELL", quantity=1), user_id=1)
    assert cur.cash == pytest.approx(50_000)


def test_unknown_symbol_has_no_price(market, monkeypatch):
    _wire(monkeypatch)
    with pytest.raises(HTTPException) as e:
        T.create_trade(_order(symbol="NOPE"), user_id=1)
    assert e.value.status_code == 503


def test_refresh_quote_pool_refills_expired_crypto_cache(monkeypatch):
    # Regression: the refresh looked for the fetchers on backend.main after they
    # moved to the market router, so every trade on an expired cache failed.
    import anyio

    from backend.routes import market as M

    async def fake_fetch():
        return [{"symbol": "BINANCE:BTCUSDT", "sub": "BTC", "price": 50_000.0}]

    monkeypatch.setattr(M, "_fetch_crypto_live", fake_fetch)
    cache.invalidate("crypto")

    async def run():
        await anyio.to_thread.run_sync(T._refresh_quote_pool, "crypto")

    anyio.run(run)
    assert T._live_price("BTC") == 50_000.0
    cache.invalidate("crypto")
