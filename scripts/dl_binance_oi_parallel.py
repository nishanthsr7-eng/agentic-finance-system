"""
Parallel Binance OPEN-INTEREST downloader (the fast path).

The stock ``dl_binance.py --oi`` fetches day-by-day, serially → ~14 min/symbol (~3 h for all 14).
This version keeps the SAME source + output format but fetches a symbol's daily metrics files
CONCURRENTLY (thread pool — the work is network-bound), cutting each symbol to well under a minute.

Design choices for safety:
  • Symbols processed one at a time (bounded memory); only the day-fetch is parallelized.
  • Per-thread requests.Session (requests Sessions aren't guaranteed thread-safe to share).
  • SKIPS any symbol whose oi_<SYM>.csv already exists (re-runnable; use --force to redo).
  • Identical columns/rows to the serial puller, so backend/.../binance_oi.py reads it unchanged.

    python scripts/dl_binance_oi_parallel.py                       # all missing symbols
    python scripts/dl_binance_oi_parallel.py --symbols BNB SOL     # subset
    python scripts/dl_binance_oi_parallel.py --workers 32 --force  # faster / redo everything
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dl_binance import OUT, PAIRS  # noqa: E402  (reuse universe + output dir)

OI_DIR = OUT / "open_interest"
BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
OI_COLS = [
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]

_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = _local.s = requests.Session()
    return s


def _days(start_year: int):
    d, end = dt.date(start_year, 1, 1), dt.date.today()
    while d <= end:
        yield d.strftime("%Y-%m-%d")
        d += dt.timedelta(days=1)


def _grab(pair: str, day: str) -> pd.DataFrame | None:
    url = f"{BASE}/{pair}/{pair}-metrics-{day}.zip"
    try:
        r = _session().get(url, timeout=60)
        if r.status_code != 200 or not r.content:
            return None
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            inner = z.read(z.namelist()[0])
        if not inner.strip():
            return None
        df = pd.read_csv(io.BytesIO(inner), header=None)
        try:
            float(str(df.iloc[0, 0]))  # numeric first cell → no header row
        except ValueError:
            df = df.iloc[1:].reset_index(drop=True)
        if df.shape[1] == len(OI_COLS):
            df.columns = OI_COLS
        return df
    except Exception:
        return None


def pull_symbol(sym: str, pair: str, start_year: int, workers: int) -> int:
    t0 = time.time()
    days = list(_days(start_year))
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_grab, pair, d): d for d in days}
        for f in as_completed(futs):
            df = f.result()
            if df is not None:
                frames.append(df)
    if not frames:
        print(f"  {sym:5} oi: no data ({time.time() - t0:.0f}s)", flush=True)
        return 0
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    OI_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OI_DIR / f"oi_{sym}.csv", index=False)
    print(
        f"  {sym:5} oi: {len(out):>7} rows in {time.time() - t0:.0f}s "
        f"({sum(1 for x in frames)} days)",
        flush=True,
    )
    return len(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", help="subset (default: all 14)")
    ap.add_argument("--start", type=int, default=2021)
    ap.add_argument("--workers", type=int, default=24, help="concurrent day fetches per symbol")
    ap.add_argument("--force", action="store_true", help="re-download even if oi_<SYM>.csv exists")
    a = ap.parse_args(argv)

    want = [s.upper() for s in (a.symbols or list(PAIRS))]
    todo = [s for s in want if a.force or not (OI_DIR / f"oi_{s}.csv").exists()]
    skip = [s for s in want if s not in todo]
    print(f"Parallel OI -> {OI_DIR}  (start {a.start}, {a.workers} workers)")
    print(f"  skip existing ({len(skip)}): {skip}")
    print(f"  fetching      ({len(todo)}): {todo}", flush=True)

    t0, total = time.time(), 0
    for sym in todo:
        total += pull_symbol(sym, PAIRS[sym], a.start, a.workers)
    print(f"DONE — {total} OI rows for {len(todo)} symbols in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
