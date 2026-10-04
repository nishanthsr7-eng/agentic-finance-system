"""MySQL connection pool: reuse, safety on errors/open transactions, size cap, off by default."""

from __future__ import annotations

import pytest
from pymysql.constants import SERVER_STATUS

from backend import mysql_db as M
from backend.config import settings


class FakeConn:
    def __init__(self):
        self.open, self.server_status, self.pings = True, 0, 0

    def close(self):
        self.open = False

    def ping(self, reconnect=True):
        self.pings += 1


@pytest.fixture
def made(monkeypatch):
    conns: list[FakeConn] = []

    def connect(**_kw):
        conns.append(FakeConn())
        return conns[-1]

    monkeypatch.setattr(M.pymysql, "connect", connect)
    monkeypatch.setattr(settings, "MYSQL_POOL_SIZE", 2)
    monkeypatch.setattr(settings, "MYSQL_POOL_PING_AFTER_S", 30)
    M.close_pool()
    yield conns
    M.close_pool()


def test_default_is_no_pool():
    assert type(settings).model_fields["MYSQL_POOL_SIZE"].default == 0


def test_pool_off_connects_and_closes_every_call(made, monkeypatch):
    monkeypatch.setattr(settings, "MYSQL_POOL_SIZE", 0)
    for _ in range(3):
        with M.get_conn():
            pass
    assert len(made) == 3 and not any(c.open for c in made)


def test_connection_is_reused(made):
    for _ in range(5):
        with M.get_conn() as conn:
            assert conn is made[0]
    assert len(made) == 1 and made[0].open
    assert made[0].pings == 0  # fresh: no ping needed


def test_idle_connection_is_pinged_before_reuse(made, monkeypatch):
    with M.get_conn():
        pass
    monkeypatch.setattr(settings, "MYSQL_POOL_PING_AFTER_S", -1)
    with M.get_conn():
        pass
    assert made[0].pings == 1


def test_dead_connection_is_replaced(made, monkeypatch):
    with M.get_conn():
        pass

    def broken(reconnect=True):
        raise OSError("gone")

    made[0].ping = broken
    monkeypatch.setattr(settings, "MYSQL_POOL_PING_AFTER_S", -1)
    with M.get_conn() as conn:
        assert conn is made[1]
    assert not made[0].open


def test_error_inside_block_closes_instead_of_pooling(made):
    with pytest.raises(RuntimeError):
        with M.get_conn():
            raise RuntimeError("query failed")
    assert not made[0].open
    with M.get_conn() as conn:
        assert conn is made[1]


def test_open_transaction_is_never_pooled(made):
    with M.get_conn() as conn:
        conn.server_status = SERVER_STATUS.SERVER_STATUS_IN_TRANS
    assert not made[0].open
    with M.get_conn() as conn:
        assert conn is made[1]


def test_pool_size_cap(made):
    with M.get_conn(), M.get_conn(), M.get_conn():
        pass  # three borrowed at once
    assert len(made) == 3
    assert sum(c.open for c in made) == 2  # only two kept


def test_admin_connection_without_db_is_not_pooled(made):
    with M.get_conn(include_db=False):
        pass
    assert not made[0].open
