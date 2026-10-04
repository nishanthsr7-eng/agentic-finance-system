"""ingestion_log retention (step 9d) and the backup script's JSON encoding."""

from __future__ import annotations

import asyncio
import datetime as dt
import decimal
import importlib.util
import time
from pathlib import Path

import aiosqlite
import pytest

from backend import db
from backend.config import settings


def _rows(path):
    async def go():
        async with aiosqlite.connect(path) as c:
            async with c.execute("SELECT job FROM ingestion_log ORDER BY job") as cur:
                return [r[0] for r in await cur.fetchall()]

    return asyncio.run(go())


def _seed(monkeypatch, tmp_path):
    path = tmp_path / "m.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    asyncio.run(db.init_db())
    now = int(time.time() * 1000)

    async def go():
        async with aiosqlite.connect(path) as c:
            for job, age_days in (("new", 1), ("edge", 29), ("old", 31), ("ancient", 400)):
                await c.execute(
                    "INSERT INTO ingestion_log (job, status, rows, message, ts) VALUES (?,?,?,?,?)",
                    (job, "ok", 0, "", now - age_days * 86_400_000),
                )
            await c.commit()

    asyncio.run(go())
    return path


def test_defaults():
    assert settings.INGESTION_LOG_RETENTION_DAYS == 30
    assert settings.SNAPSHOT_RETENTION_DAYS == 7


def test_prune_ingestion_log_keeps_recent(monkeypatch, tmp_path):
    path = _seed(monkeypatch, tmp_path)
    assert asyncio.run(db.prune_ingestion_log(30)) == 2
    assert _rows(path) == ["edge", "new"]


def test_prune_zero_keeps_everything(monkeypatch, tmp_path):
    path = _seed(monkeypatch, tmp_path)
    assert asyncio.run(db.prune_ingestion_log(0)) == 0
    assert len(_rows(path)) == 4


def test_backup_encodes_mysql_types():
    path = Path(__file__).parents[2] / "scripts" / "db_backup.py"
    if not path.exists():  # the Docker image ships backend/ only
        pytest.skip("scripts/ not present")
    spec = importlib.util.spec_from_file_location("db_backup", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._jsonable(dt.datetime(2026, 10, 4, 5, 6)) == "2026-10-04T05:06:00"
    assert mod._jsonable(dt.date(2026, 10, 4)) == "2026-10-04"
    assert mod._jsonable(decimal.Decimal("12.50")) == "12.50"
