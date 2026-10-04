"""Migration runner: order, once-only, failure handling, and the shipped files."""

from __future__ import annotations

import pytest

from backend import migrate


class Cur:
    """Records SQL; answers SELECT version / SHOW COLUMNS from in-memory state."""

    def __init__(self, applied=(), columns=(), fail_on=None):
        self.sql, self.applied, self.columns, self.fail_on = [], set(applied), set(columns), fail_on
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        if self.fail_on and self.fail_on in sql:
            raise RuntimeError("boom")
        self.sql.append(sql)
        if sql.startswith("SELECT version"):
            self._rows = [{"version": v} for v in self.applied]
        elif sql.startswith("SHOW COLUMNS"):
            table = sql.split("`")[1]
            self._rows = [{"Field": params[0]}] if (table, params[0]) in self.columns else []
        elif sql.startswith("INSERT INTO schema_migrations"):
            self.applied.add(params[0])
        else:
            self._rows = []

    def fetchall(self):
        return self._rows


class Conn:
    def __init__(self, cur):
        self.cur = cur

    def cursor(self):
        return self.cur


def _write(tmp_path, files):
    for name, body in files.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    return tmp_path


def test_applies_pending_in_order_once(tmp_path):
    d = _write(
        tmp_path,
        {
            "002_b.sql": "CREATE TABLE b (x INT);",
            "001_a.sql": "-- first\nCREATE TABLE a (x INT);\nCREATE TABLE a2 (x INT);",
            "notes.txt": "ignored",
        },
    )
    cur = Cur()
    assert migrate.run_migrations(Conn(cur), d) == ["001_a.sql", "002_b.sql"]
    creates = [s for s in cur.sql if s.startswith("CREATE TABLE ") and "schema_migrations" not in s]
    assert creates == [
        "CREATE TABLE a (x INT)",
        "CREATE TABLE a2 (x INT)",
        "CREATE TABLE b (x INT)",
    ]
    assert migrate.run_migrations(Conn(cur), d) == []  # second boot: nothing to do


def test_failure_stops_and_is_not_recorded(tmp_path):
    d = _write(
        tmp_path,
        {
            "001_a.sql": "CREATE TABLE a (x INT)",
            "002_b.sql": "CREATE TABLE bad (x INT)",
            "003_c.sql": "CREATE TABLE c (x INT)",
        },
    )
    cur = Cur(fail_on="bad")
    with pytest.raises(RuntimeError):
        migrate.run_migrations(Conn(cur), d)
    assert cur.applied == {1}
    assert not any("TABLE c" in s for s in cur.sql)


def test_duplicate_versions_are_rejected(tmp_path):
    d = _write(tmp_path, {"001_a.sql": "", "001_b.sql": ""})
    with pytest.raises(RuntimeError, match="duplicate"):
        migrate.discover(d)


def test_python_migration_and_add_column_if_missing(tmp_path):
    d = _write(
        tmp_path,
        {
            "001_col.py": (
                "from backend.migrate import add_column_if_missing\n"
                "def up(cur):\n"
                "    add_column_if_missing(cur, 'users', 'password_hash', 'VARCHAR(255) NULL')\n"
                "    add_column_if_missing(cur, 'trades', 'asset_type', 'VARCHAR(20)')\n"
            )
        },
    )
    cur = Cur(columns={("users", "password_hash")})  # already there on live
    migrate.run_migrations(Conn(cur), d)
    alters = [s for s in cur.sql if s.startswith("ALTER")]
    assert alters == ["ALTER TABLE `trades` ADD COLUMN `asset_type` VARCHAR(20)"]


def test_shipped_migrations_are_noops_on_the_live_schema():
    """On a DB that already has every column, the real files add nothing."""
    live = {
        ("users", "password_hash"),
        ("trades", "asset_type"),
        ("accounts", "credit_limit"),
        ("predictions", "iv_atm"),
        ("predictions", "iv_skew"),
        ("predictions", "conf_low_90"),
        ("predictions", "conf_high_90"),
        ("prediction_outcomes", "in_band"),
    }
    cur = Cur(columns=live)
    names = migrate.run_migrations(Conn(cur))
    assert names == [p.name for _v, p in migrate.discover()]
    assert [v for v, _p in migrate.discover()] == list(range(1, len(names) + 1))
    assert not any(
        s.strip().startswith(("ALTER", "DROP", "DELETE", "UPDATE", "TRUNCATE")) for s in cur.sql
    )
    assert all(
        s.strip().startswith(
            (
                "CREATE TABLE IF NOT EXISTS",
                "SHOW COLUMNS",
                "SELECT version",
                "INSERT INTO schema_migrations",
                "SHOW INDEX",
                "CREATE INDEX",
            )
        )
        for s in cur.sql
    )


def test_startup_wrapper_never_raises(monkeypatch):
    def down(*_a, **_k):
        raise OSError("MySQL unreachable")

    monkeypatch.setattr(migrate, "run_migrations", down)
    migrate.run_migrations_safely()
