"""/db/bootstrap: same payloads as the 7 separate routes, over one connection."""

from __future__ import annotations

import pytest

from backend import mysql_db as M
from backend import user_api as U
from backend.config import settings


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        self.conn.sql.append(sql)
        self.table = sql.split(" FROM ")[1].split()[0]

    def fetchall(self):
        return [{"table": self.table}]

    def fetchone(self):
        return {"table": self.table}


class FakeConn:
    def __init__(self):
        self.sql: list[str] = []

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        pass


@pytest.fixture
def made(monkeypatch):
    conns: list[FakeConn] = []

    def connect(**_kw):
        conns.append(FakeConn())
        return conns[-1]

    monkeypatch.setattr(M.pymysql, "connect", connect)
    monkeypatch.setattr(settings, "MYSQL_POOL_SIZE", 0)
    return conns


def test_bootstrap_uses_one_connection(made):
    out = U.get_bootstrap(user_id=1, tx_limit=2000)
    assert len(made) == 1
    # tx, accounts, 2x portfolio, recurring, contacts, 3x security, rewards
    assert len(made[0].sql) == 10
    assert set(out) == {
        "transactions",
        "accounts",
        "portfolio",
        "recurring",
        "contacts",
        "security",
        "rewards",
    }


def test_bootstrap_matches_separate_routes(made):
    out = U.get_bootstrap(user_id=1, tx_limit=2000)
    assert out["transactions"] == U.get_transactions(user_id=1, limit=2000, category=None)
    assert out["accounts"] == U.get_accounts(user_id=1)
    assert out["portfolio"] == U.get_portfolio(user_id=1)
    assert out["recurring"] == U.get_recurring(user_id=1)
    assert out["contacts"] == U.get_contacts(user_id=1)
    assert out["security"] == U.get_security(user_id=1)
    assert out["rewards"] == U.get_rewards(user_id=1)
