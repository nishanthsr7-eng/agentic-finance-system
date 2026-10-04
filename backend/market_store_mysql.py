"""
FLUX — market store on MySQL / TiDB
===================================

Sync TiDB versions of the backend.db functions for the four tables that must
survive a Render restart: predictions, prediction_outcomes,
calibration_buckets and ohlcv_history. backend.db hands off to these through
asyncio.to_thread when settings.MARKET_STORE == "mysql"; every other market
table stays in SQLite (a cache that refills itself).

The tables themselves come from the migrations (001 baseline, 007 indexes, 008 range).
Same names, columns and return shapes as the SQLite versions.
"""

from __future__ import annotations

from . import mysql_db
from .mysql_db import get_conn


def query(sql: str, params: tuple | None = None) -> list[dict]:
    """mysql_db.query, always a list (PyMySQL returns () for no rows) like SQLite."""
    return list(mysql_db.query(sql, params))


# executemany chunk: one round trip per chunk keeps TiDB Request Units low
# without building a huge statement.
_BATCH = 500

_PRED_COLS = (
    "symbol",
    "model",
    "horizon_days",
    "direction",
    "prob_up",
    "meta_prob",
    "act",
    "confidence",
    "kelly_frac",
    "sentiment",
    "last_close",
    "pred_return",
    "pred_price",
    "conf_low",
    "conf_high",
    "regime",
    "generated_at",
    "target_date",
    "iv_atm",
    "iv_skew",
    "conf_low_90",
    "conf_high_90",
)
_HIST_COLS = ("symbol", "date", "open", "high", "low", "close", "adj_close", "volume")
_OUT_COLS = (
    "prediction_id",
    "actual_return",
    "correct",
    "pnl_after_costs",
    "resolved_at",
    "in_band",
)
_CAL_COLS = ("model", "bucket", "stated_conf", "realized_hit", "n", "updated_at")


def _upsert_sql(table: str, cols: tuple[str, ...], key: tuple[str, ...]) -> str:
    names = ",".join(f"`{c}`" for c in cols)
    marks = ",".join(["%s"] * len(cols))
    upd = ",".join(f"`{c}`=VALUES(`{c}`)" for c in cols if c not in key)
    return f"INSERT INTO {table} ({names}) VALUES ({marks}) ON DUPLICATE KEY UPDATE {upd}"


def _executemany(sql: str, cols: tuple[str, ...], rows: list[dict]) -> None:
    if not rows:
        return
    values = [tuple(r.get(c) for c in cols) for r in rows]
    with get_conn() as conn, conn.cursor() as cur:
        for i in range(0, len(values), _BATCH):
            cur.executemany(sql, values[i : i + _BATCH])


# ── writes ───────────────────────────────────────────────────────────────────


def insert_history(rows: list[dict]) -> None:
    _executemany(_upsert_sql("ohlcv_history", _HIST_COLS, ("symbol", "date")), _HIST_COLS, rows)


def insert_prediction(row: dict) -> int:
    names = ",".join(f"`{c}`" for c in _PRED_COLS)
    marks = ",".join(["%s"] * len(_PRED_COLS))
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO predictions ({names}) VALUES ({marks})",
            tuple(row.get(c) for c in _PRED_COLS),
        )
        return cur.lastrowid


def insert_outcome(row: dict) -> None:
    _executemany(
        _upsert_sql("prediction_outcomes", _OUT_COLS, ("prediction_id",)), _OUT_COLS, [row]
    )


def upsert_calibration(rows: list[dict]) -> None:
    _executemany(
        _upsert_sql("calibration_buckets", _CAL_COLS, ("model", "bucket")), _CAL_COLS, rows
    )


# ── reads ────────────────────────────────────────────────────────────────────


def get_due_predictions(as_of_date: str) -> list[dict]:
    return query(
        "SELECT p.* FROM predictions p "
        "LEFT JOIN prediction_outcomes o ON p.id = o.prediction_id "
        "WHERE o.prediction_id IS NULL AND p.target_date IS NOT NULL "
        "AND p.target_date <= %s ORDER BY p.target_date ASC",
        (as_of_date,),
    )


def get_latest_predictions(symbol: str | None = None, limit: int = 50) -> list[dict]:
    if symbol:
        return query(
            "SELECT * FROM predictions WHERE symbol=%s ORDER BY generated_at DESC LIMIT %s",
            (symbol.upper(), limit),
        )
    return query(
        "SELECT p.* FROM predictions p JOIN ("
        "  SELECT symbol, MAX(generated_at) AS g FROM predictions GROUP BY symbol"
        ") m ON p.symbol=m.symbol AND p.generated_at=m.g ORDER BY p.confidence DESC LIMIT %s",
        (limit,),
    )


def get_prediction_history(symbol: str, limit: int = 100) -> list[dict]:
    return query(
        "SELECT p.*, o.actual_return, o.correct, o.pnl_after_costs, o.resolved_at, o.in_band "
        "FROM predictions p LEFT JOIN prediction_outcomes o ON p.id=o.prediction_id "
        "WHERE p.symbol=%s ORDER BY p.generated_at DESC LIMIT %s",
        (symbol.upper(), limit),
    )


def get_resolved_outcomes() -> list[dict]:
    return query(
        "SELECT p.confidence, p.meta_prob, o.correct, o.actual_return, o.pnl_after_costs "
        "FROM prediction_outcomes o JOIN predictions p ON p.id=o.prediction_id"
    )


def get_calibration_buckets(model: str = "live") -> list[dict]:
    return query("SELECT * FROM calibration_buckets WHERE model=%s ORDER BY bucket ASC", (model,))


def get_history(symbol: str, start: str | None = None) -> list[dict]:
    if start:
        return query(
            "SELECT * FROM ohlcv_history WHERE symbol=%s AND `date`>=%s ORDER BY `date` ASC",
            (symbol.upper(), start),
        )
    return query(
        "SELECT * FROM ohlcv_history WHERE symbol=%s ORDER BY `date` ASC", (symbol.upper(),)
    )


def history_summary() -> list[dict]:
    return query(
        "SELECT symbol, COUNT(*) AS `rows`, MIN(`date`) AS `first`, MAX(`date`) AS `last` "
        "FROM ohlcv_history GROUP BY symbol ORDER BY symbol"
    )


def get_closes(symbols: list[str], start: str, end: str) -> list[dict]:
    """symbol/date/close rows for the demo seed's trade prices."""
    if not symbols:
        return []
    marks = ",".join(["%s"] * len(symbols))
    return query(
        f"SELECT symbol, `date`, close FROM ohlcv_history "
        f"WHERE symbol IN ({marks}) AND `date` BETWEEN %s AND %s",
        (*symbols, start, end),
    )
