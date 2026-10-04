"""
One-off: copy ohlcv_history from the local SQLite DB into MySQL/TiDB, for the
switch to MARKET_STORE=mysql (step 9c). Run it from your own machine with the
TiDB credentials in .env; Render never runs it.

    python scripts/copy_history_to_tidb.py                 # dry run: counts only
    python scripts/copy_history_to_tidb.py --apply         # upsert into TiDB

Only rows on/after --since are copied (default 2020-01-01). Inference needs a
few hundred bars per symbol, and a bounded window keeps both the TiDB quota and
the 00:30 prediction job's memory on Render small. The upsert is idempotent, so
running it twice is harmless. Predictions/outcomes are NOT copied: local ones
come from dev runs, and the live track record should start from live runs.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import market_store_mysql as ms  # noqa: E402
from backend.db import DB_PATH  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", default="2020-01-01", help="first date to copy (YYYY-MM-DD)")
    ap.add_argument("--apply", action="store_true", help="write to TiDB (default: dry run)")
    args = ap.parse_args()

    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row
        rows = [
            dict(r)
            for r in db.execute(
                "SELECT * FROM ohlcv_history WHERE date >= ? ORDER BY symbol, date", (args.since,)
            )
        ]
    by_sym: dict[str, int] = {}
    for r in rows:
        by_sym[r["symbol"]] = by_sym.get(r["symbol"], 0) + 1
    print(f"{len(rows)} rows, {len(by_sym)} symbols since {args.since} in {DB_PATH}")
    for s, n in sorted(by_sym.items()):
        print(f"  {s:10s} {n}")

    if not args.apply:
        print("dry run: nothing written (add --apply)")
        return
    ms.insert_history(rows)
    print("TiDB now:", len(ms.history_summary()), "symbols")


if __name__ == "__main__":
    main()
