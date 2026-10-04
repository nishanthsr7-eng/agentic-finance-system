"""MARKET_STORE switch: default stays SQLite, "mysql" routes the four
persistent tables to backend.market_store_mysql, plus migration 007."""

from __future__ import annotations

import asyncio
import importlib.util
from datetime import date
from pathlib import Path

import pytest

from backend import db, demo_seed
from backend import market_store_mysql as ms
from backend.config import settings

ROUTED = {
    "insert_history": ([{"symbol": "BTC"}],),
    "insert_prediction": ({"symbol": "BTC"},),
    "insert_outcome": ({"prediction_id": 1},),
    "get_due_predictions": ("2026-10-04",),
    "get_latest_predictions": ("BTC", 5),
    "get_prediction_history": ("BTC", 5),
    "get_resolved_outcomes": (),
    "get_calibration_buckets": ("live",),
    "upsert_calibration": ([{"model": "live", "bucket": 5}],),
    "get_history": ("BTC", "2026-01-01"),
    "history_summary": (),
}


class Cur:
    def __init__(self, rows=()):
        self.sql, self.many, self.rows, self.lastrowid = [], [], list(rows), 42

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        self.sql.append((sql, params))

    def executemany(self, sql, values):
        self.many.append((sql, list(values)))

    def fetchall(self):
        return self.rows


class Conn:
    def __init__(self, cur):
        self.cur = cur

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self.cur


@pytest.fixture
def mysql_store(monkeypatch):
    monkeypatch.setattr(settings, "MARKET_STORE", "mysql")


def test_default_is_sqlite():
    assert settings.MARKET_STORE == "sqlite"
    assert not db._mysql_store()


@pytest.mark.parametrize("name", sorted(ROUTED))
def test_mysql_store_routes_every_function(mysql_store, monkeypatch, name):
    calls = []
    monkeypatch.setattr(ms, name, lambda *a: calls.append(a) or "from-tidb")
    out = asyncio.run(getattr(db, name)(*ROUTED[name]))
    assert out == "from-tidb"
    assert calls == [ROUTED[name]]


def test_sqlite_store_never_touches_mysql(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "m.db")
    monkeypatch.setattr(db, "_ms", lambda: pytest.fail("mysql store used"))
    asyncio.run(db.init_db())
    asyncio.run(
        db.insert_history(
            [
                {
                    "symbol": "BTC",
                    "date": "2026-10-01",
                    "open": 1,
                    "high": 1,
                    "low": 1,
                    "close": 1,
                    "adj_close": 1,
                    "volume": 1,
                }
            ]
        )
    )
    assert asyncio.run(db.history_summary())[0]["rows"] == 1


def test_history_upsert_is_batched(monkeypatch):
    cur = Cur()
    monkeypatch.setattr(ms, "get_conn", lambda: Conn(cur))
    rows = [{"symbol": "BTC", "date": f"d{i}", "close": i} for i in range(1200)]
    ms.insert_history(rows)
    assert [len(v) for _, v in cur.many] == [500, 500, 200]
    sql = cur.many[0][0]
    assert sql.startswith("INSERT INTO ohlcv_history") and "ON DUPLICATE KEY UPDATE" in sql
    assert "`symbol`=VALUES" not in sql and "`close`=VALUES(`close`)" in sql
    assert cur.many[0][1][3][1] == "d3"  # column order symbol, date, ...


def test_insert_prediction_returns_id_and_fills_missing(monkeypatch):
    cur = Cur()
    monkeypatch.setattr(ms, "get_conn", lambda: Conn(cur))
    assert ms.insert_prediction({"symbol": "BTC", "model": "m"}) == 42
    sql, params = cur.sql[0]
    assert len(params) == len(ms._PRED_COLS) and params[0] == "BTC" and params[-1] is None


def test_empty_writes_skip_the_db(monkeypatch):
    monkeypatch.setattr(ms, "get_conn", lambda: pytest.fail("connected"))
    ms.insert_history([])
    ms.upsert_calibration([])
    assert ms.get_closes([], "a", "b") == []


def test_demo_seed_reads_closes_from_tidb(mysql_store, monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "missing.db")
    seen = []

    def fake(syms, lo, hi):
        seen.append((syms, lo, hi))
        return [{"symbol": "AAPL", "date": "2026-10-01", "close": 250.5}]

    monkeypatch.setattr(ms, "get_closes", fake)
    out = demo_seed.load_closes(date(2026, 9, 10), date(2026, 10, 4))
    assert out == {"AAPL": {"2026-10-01": 250.5}}
    assert seen[0][1:] == ("2026-09-03", "2026-10-04")


def _migration_007():
    path = Path(__file__).parents[1] / "migrations" / "007_market_store_indexes.py"
    spec = importlib.util.spec_from_file_location("m007", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class IndexCur(Cur):
    def __init__(self, indexes, columns):
        super().__init__()
        self.indexes, self.columns = set(indexes), set(columns)

    def execute(self, sql, params=()):
        super().execute(sql, params)
        if sql.startswith("SHOW INDEX"):
            self.rows = (
                [{"Key_name": params[0]}] if (sql.split("`")[1], params[0]) in self.indexes else []
            )
        elif sql.startswith("SHOW COLUMNS"):
            self.rows = (
                [{"Field": params[0]}] if (sql.split("`")[1], params[0]) in self.columns else []
            )
        else:
            self.rows = []


def test_migration_007_creates_only_missing():
    m = _migration_007()
    have = {(t, n) for t, n, _ in m.INDEXES if n != "idx_pred_target"}
    cur = IndexCur(have, {("predictions", "iv_atm"), ("predictions", "iv_skew")})
    m.up(cur)
    changes = [s for s, _ in cur.sql if s.startswith(("CREATE", "ALTER"))]
    assert changes == ["CREATE INDEX `idx_pred_target` ON `predictions` (target_date)"]


def test_migration_007_on_a_bare_schema():
    m = _migration_007()
    cur = IndexCur(set(), set())
    m.up(cur)
    changes = [s for s, _ in cur.sql if s.startswith(("CREATE", "ALTER"))]
    assert len(changes) == 2 + len(m.INDEXES)
