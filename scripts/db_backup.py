"""
Back up / restore the MySQL (TiDB) database from your own machine.

Pure Python over PyMySQL (already a dependency), so it needs no mysqldump.
Reads the connection from .env like the app does. Run it with the ADMIN user.

    python scripts/db_backup.py                       # back up every table
    python scripts/db_backup.py --out D:/backups      # choose the folder
    python scripts/db_backup.py --restore FILE        # dry run: show what's in FILE
    python scripts/db_backup.py --restore FILE --apply --tables faqs users

A backup is one gzipped JSON file: {"meta": {...}, "tables": {name: [rows]}}.
It contains password hashes and personal demo data: keep it out of git
(the default folder is outside the repo) and out of shared drives.

Restore never drops tables. It upserts rows (INSERT ... ON DUPLICATE KEY
UPDATE) into tables that already exist, so run the migrations first on an
empty database (python -m backend.migrate). See scripts/db_restore.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import decimal
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings  # noqa: E402
from backend.mysql_db import get_conn  # noqa: E402

DEFAULT_DIR = Path.home() / "flux-backups"
_BATCH = 500


def _jsonable(v):
    if isinstance(v, (dt.datetime, dt.date, dt.time)):
        return v.isoformat()
    if isinstance(v, dt.timedelta):
        return str(v)
    if isinstance(v, decimal.Decimal):
        return str(v)
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    raise TypeError(type(v))


def backup(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{settings.MYSQL_DB}-{stamp}.json.gz"
    data: dict = {
        "meta": {
            "db": settings.MYSQL_DB,
            "host": settings.MYSQL_HOST,
            "taken_at": dt.datetime.now().isoformat(),
        },
        "tables": {},
    }
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SHOW FULL TABLES WHERE Table_type = 'BASE TABLE'")
        tables = sorted(next(iter(r.values())) for r in cur.fetchall())
        for t in tables:
            cur.execute(f"SELECT * FROM `{t}`")
            data["tables"][t] = list(cur.fetchall())
            print(f"  {t:28s} {len(data['tables'][t])}")
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f, default=_jsonable)
    print(f"wrote {path} ({path.stat().st_size // 1024} KB)")
    return path


def restore(path: Path, only: list[str] | None, apply: bool) -> None:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    print("backup of", data["meta"])
    tables = {t: rows for t, rows in data["tables"].items() if not only or t in only}
    for t, rows in tables.items():
        print(f"  {t:28s} {len(rows)}")
    if not apply:
        print("dry run: nothing written (add --apply)")
        return
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SET FOREIGN_KEY_CHECKS=0")
        try:
            for t, rows in tables.items():
                if not rows:
                    continue
                cols = list(rows[0])
                names = ",".join(f"`{c}`" for c in cols)
                marks = ",".join(["%s"] * len(cols))
                upd = ",".join(f"`{c}`=VALUES(`{c}`)" for c in cols)
                sql = f"INSERT INTO `{t}` ({names}) VALUES ({marks}) ON DUPLICATE KEY UPDATE {upd}"
                values = [tuple(r.get(c) for c in cols) for r in rows]
                for i in range(0, len(values), _BATCH):
                    cur.executemany(sql, values[i : i + _BATCH])
                print(f"  restored {t}")
        finally:
            cur.execute("SET FOREIGN_KEY_CHECKS=1")


def main() -> None:
    ap = argparse.ArgumentParser(description="Back up / restore the MySQL database.")
    ap.add_argument(
        "--out", type=Path, default=DEFAULT_DIR, help=f"backup folder (default {DEFAULT_DIR})"
    )
    ap.add_argument("--restore", type=Path, help="backup file to restore from")
    ap.add_argument("--tables", nargs="*", help="restore only these tables")
    ap.add_argument("--apply", action="store_true", help="with --restore: actually write")
    args = ap.parse_args()
    if args.restore:
        restore(args.restore, args.tables, args.apply)
    else:
        backup(args.out)


if __name__ == "__main__":
    main()
