"""
FLUX — Schema migrations
========================

Files in backend/migrations/ named NNN_description.sql or NNN_description.py
are applied in version order, once each, and recorded in `schema_migrations`.

  .sql  statements separated by ';' (full-line `--` comments allowed)
  .py   a module with `def up(cur)`, for changes that need a check first,
        e.g. add_column_if_missing (MySQL has no ADD COLUMN IF NOT EXISTS)

MySQL commits DDL implicitly, so a migration can't be rolled back halfway.
Every migration must therefore be safe to run again: CREATE ... IF NOT EXISTS,
add_column_if_missing. If one fails it isn't recorded, and the next boot
retries it.

Startup calls run_migrations(). By hand:

    python -m backend.migrate --dry-run   # list pending, change nothing
    python -m backend.migrate             # apply pending
"""

from __future__ import annotations

import importlib.util
import logging
import re
import sys
from pathlib import Path

log = logging.getLogger("flux.migrate")

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_NAME = re.compile(r"^(\d{3})_[a-z0-9_]+\.(sql|py)$")

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    INT PRIMARY KEY,
    name       VARCHAR(120) NOT NULL,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def discover(directory: Path = MIGRATIONS_DIR) -> list[tuple[int, Path]]:
    """(version, path) for every migration file, sorted. Duplicate versions are an error."""
    found: dict[int, Path] = {}
    for p in sorted(directory.iterdir()):
        m = _NAME.match(p.name)
        if not m:
            continue
        v = int(m.group(1))
        if v in found:
            raise RuntimeError(f"duplicate migration version {v:03d}: {found[v].name}, {p.name}")
        found[v] = p
    return sorted(found.items())


def split_sql(text: str) -> list[str]:
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def add_column_if_missing(cur, table: str, column: str, definition: str) -> bool:
    cur.execute(f"SHOW COLUMNS FROM `{table}` LIKE %s", (column,))
    if cur.fetchall():
        return False
    cur.execute(f"ALTER TABLE `{table}` ADD COLUMN `{column}` {definition}")
    log.info("migration: %s.%s added", table, column)
    return True


def _apply(cur, path: Path) -> None:
    if path.suffix == ".sql":
        for stmt in split_sql(path.read_text(encoding="utf-8")):
            cur.execute(stmt)
        return
    spec = importlib.util.spec_from_file_location(f"flux_migration_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load migration {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.up(cur)


def pending(cur, directory: Path = MIGRATIONS_DIR) -> list[tuple[int, Path]]:
    cur.execute(_TABLE_DDL)
    cur.execute("SELECT version FROM schema_migrations")
    done = {r["version"] for r in cur.fetchall()}
    return [(v, p) for v, p in discover(directory) if v not in done]


def run_migrations(conn=None, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply every pending migration in order; return the names applied.
    Stops at the first failure (later ones depend on earlier ones)."""
    if conn is None:
        from . import mysql_db as M

        with M.get_conn() as c:
            return run_migrations(c, directory)

    applied = []
    with conn.cursor() as cur:
        for version, path in pending(cur, directory):
            _apply(cur, path)
            cur.execute(
                "INSERT INTO schema_migrations (version, name) VALUES (%s, %s)",
                (version, path.name),
            )
            applied.append(path.name)
            log.info("migration applied: %s", path.name)
    if not applied:
        log.info("schema up to date")
    return applied


def run_migrations_safely() -> None:
    """Startup wrapper: a database outage must not stop the API from booting."""
    try:
        run_migrations()
    except Exception as e:  # noqa: BLE001 — MySQL down / bad migration: routes surface it
        log.warning("migrations skipped: %s", e)


def _dry_run() -> None:
    from . import mysql_db as M

    with M.get_conn() as conn, conn.cursor() as cur:
        cur.execute("SHOW TABLES LIKE 'schema_migrations'")
        if cur.fetchall():
            cur.execute("SELECT version FROM schema_migrations")
            done = {r["version"] for r in cur.fetchall()}
        else:
            done = set()
            print("schema_migrations table does not exist yet (created on first run)")
        for v, p in discover():
            print(f"  {'applied' if v in done else 'PENDING'}  {p.name}")
        # What the column migrations would actually change (read-only).
        for table, col in (
            ("users", "password_hash"),
            ("trades", "asset_type"),
            ("accounts", "credit_limit"),
        ):
            cur.execute(f"SHOW COLUMNS FROM `{table}` LIKE %s", (col,))
            print(f"  {table}.{col}: {'present' if cur.fetchall() else 'MISSING (would be added)'}")
        for table in ("trading_wallet", "watchlist", "market_alerts", "app_state"):
            cur.execute("SHOW TABLES LIKE %s", (table,))
            print(
                f"  table {table}: {'present' if cur.fetchall() else 'MISSING (would be created)'}"
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "--dry-run" in sys.argv:
        _dry_run()
    else:
        print("applied:", run_migrations() or "nothing")
