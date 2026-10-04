"""
FLUX — SQLite persistence layer
Tables: price_snapshots, ohlcv_daily, ai_insights, news_cache
"""

import asyncio
import logging
import os
from pathlib import Path

import aiosqlite

from .config import settings

log = logging.getLogger("flux.db")


def _resolve_db_path() -> Path:
    """DB lives OUTSIDE the project root by default: the dev static server
    serves the project directory, so a DB kept there is publicly downloadable
    (and its WAL writes used to retrigger live-reload). Override with DB_PATH
    in .env if needed."""
    if settings.DB_PATH:
        return Path(settings.DB_PATH)
    base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / ".local" / "share")) / "flux"
    base.mkdir(parents=True, exist_ok=True)
    return base / "flux_market.db"


DB_PATH = _resolve_db_path()


def _mysql_store() -> bool:
    """MARKET_STORE=mysql keeps predictions, outcomes, calibration and
    ohlcv_history in TiDB so they survive a restart. Default: SQLite."""
    return settings.MARKET_STORE.strip().lower() == "mysql"


def _ms():
    from . import market_store_mysql

    return market_store_mysql


_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS price_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    asset_type  TEXT    NOT NULL,
    name        TEXT,
    price       REAL    NOT NULL,
    change_pct  REAL,
    volume      REAL,
    market_cap  REAL,
    ts          INTEGER NOT NULL       -- unix ms
);
CREATE INDEX IF NOT EXISTS idx_ps_symbol_ts ON price_snapshots(symbol, ts DESC);
CREATE INDEX IF NOT EXISTS idx_ps_ts        ON price_snapshots(ts DESC);

CREATE TABLE IF NOT EXISTS ohlcv_daily (
    symbol  TEXT NOT NULL,
    date    TEXT NOT NULL,             -- YYYY-MM-DD
    open    REAL,
    high    REAL,
    low     REAL,
    close   REAL,
    volume  INTEGER,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON ohlcv_daily(date DESC);

-- Long historical OHLCV for model training (years of data, separate from the
-- 30-day ohlcv_daily so the live ingestion job never overwrites it).
CREATE TABLE IF NOT EXISTS ohlcv_history (
    symbol    TEXT NOT NULL,
    date      TEXT NOT NULL,           -- YYYY-MM-DD
    open      REAL,
    high      REAL,
    low       REAL,
    close     REAL,
    adj_close REAL,                    -- split/dividend adjusted (= close for crypto)
    volume    INTEGER,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_hist_sym_date ON ohlcv_history(symbol, date DESC);

CREATE TABLE IF NOT EXISTS ai_insights (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT    NOT NULL,
    insight_type TEXT    NOT NULL,
    content      TEXT    NOT NULL,
    sentiment    TEXT,
    confidence   INTEGER,
    generated_at INTEGER NOT NULL      -- unix ms
);
CREATE INDEX IF NOT EXISTS idx_ins_ts     ON ai_insights(generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_ins_symbol ON ai_insights(symbol, generated_at DESC);

CREATE TABLE IF NOT EXISTS news_cache (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT,
    source       TEXT,
    url          TEXT UNIQUE,
    summary      TEXT,
    published_at TEXT,
    cached_at    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_news_cached ON news_cache(cached_at DESC);

-- Model predictions (serving + backtest read these).
CREATE TABLE IF NOT EXISTS predictions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT NOT NULL,
    model        TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    direction    TEXT,
    prob_up      REAL,
    meta_prob    REAL,
    act          INTEGER,
    confidence   INTEGER,
    kelly_frac   REAL,
    sentiment    REAL,
    last_close   REAL,
    pred_return  REAL,                  -- Layer-2b point return forecast
    pred_price   REAL,                  -- last_close · (1 + pred_return)
    conf_low     REAL,                  -- conformal band lower price bound
    conf_high    REAL,                  -- conformal band upper price bound
    conf_low_90  REAL,                  -- 90% band lower price bound
    conf_high_90 REAL,                  -- 90% band upper price bound
    regime       TEXT,                  -- HMM market state at prediction time
    generated_at INTEGER NOT NULL,
    target_date  TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_sym ON predictions(symbol, generated_at DESC);

-- Resolved outcomes for accuracy/calibration tracking.
CREATE TABLE IF NOT EXISTS prediction_outcomes (
    prediction_id INTEGER PRIMARY KEY,
    actual_return REAL,
    correct       INTEGER,
    pnl_after_costs REAL,
    resolved_at   INTEGER,
    in_band       INTEGER               -- 1 if the close landed inside [conf_low, conf_high]
);

-- Per-confidence-bucket realized hit-rate (refreshed by the backtest).
CREATE TABLE IF NOT EXISTS calibration_buckets (
    model        TEXT NOT NULL,
    bucket       INTEGER NOT NULL,
    stated_conf  REAL,
    realized_hit REAL,
    n            INTEGER,
    updated_at   INTEGER,
    PRIMARY KEY (model, bucket)
);

-- FinBERT sentiment scores for news (real-time Layer-2c signal).
CREATE TABLE IF NOT EXISTS news_sentiment (
    url        TEXT PRIMARY KEY,        -- FK to news_cache.url (or finnhub article url)
    symbol     TEXT,                    -- ticker this article maps to ('' = general market)
    source     TEXT,                    -- 'newsapi' | 'finnhub'
    title      TEXT,
    score      REAL,                    -- -1..+1 (signed FinBERT confidence)
    label      TEXT,                    -- positive | negative | neutral
    published_at TEXT,
    scored_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sent_symbol ON news_sentiment(symbol, scored_at DESC);

CREATE TABLE IF NOT EXISTS options_iv (
    symbol      TEXT NOT NULL,
    date        TEXT NOT NULL,        -- snapshot date (YYYY-MM-DD, UTC)
    spot        REAL,
    atm_iv      REAL,                 -- ~30d ATM implied vol (annualized)
    skew        REAL,                 -- risk-reversal proxy: OTM put IV − OTM call IV (>0 = downside fear)
    term_slope  REAL,                 -- far-tenor ATM IV − near-tenor ATM IV (>0 = contango)
    n_contracts INTEGER,
    snapshot_at INTEGER NOT NULL,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_opt_symbol ON options_iv(symbol, date DESC);

CREATE TABLE IF NOT EXISTS ingestion_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    job     TEXT    NOT NULL,
    status  TEXT    NOT NULL,   -- 'ok' | 'error'
    rows    INTEGER DEFAULT 0,
    message TEXT,
    ts      INTEGER NOT NULL
);
"""


# Columns added after the predictions table first shipped — applied idempotently on init so
# an existing flux_market.db gains the conformal-band fields without a manual migration.
_PREDICTION_MIGRATIONS = {
    "pred_return": "REAL",
    "pred_price": "REAL",
    "conf_low": "REAL",
    "conf_high": "REAL",
    "regime": "TEXT",
    "iv_atm": "REAL",
    "iv_skew": "REAL",  # live options context logged for future feature use
    "conf_low_90": "REAL",
    "conf_high_90": "REAL",
}
_OUTCOME_MIGRATIONS = {"in_band": "INTEGER"}


async def init_db() -> None:
    """Create all tables on first run, then apply additive column migrations."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(_SCHEMA)
        async with db.execute("PRAGMA table_info(predictions)") as cur:
            existing = {r[1] for r in await cur.fetchall()}
        for col, typ in _PREDICTION_MIGRATIONS.items():
            if col not in existing:
                await db.execute(f"ALTER TABLE predictions ADD COLUMN {col} {typ}")
        async with db.execute("PRAGMA table_info(prediction_outcomes)") as cur:
            existing = {r[1] for r in await cur.fetchall()}
        for col, typ in _OUTCOME_MIGRATIONS.items():
            if col not in existing:
                await db.execute(f"ALTER TABLE prediction_outcomes ADD COLUMN {col} {typ}")
        await db.commit()
    log.info("SQLite initialised at %s", DB_PATH)


# ── Writes ────────────────────────────────────────────────────────────────────


async def insert_snapshots(rows: list[dict]) -> None:
    if not rows:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            "INSERT INTO price_snapshots "
            "(symbol, asset_type, name, price, change_pct, volume, market_cap, ts) "
            "VALUES (:symbol, :asset_type, :name, :price, :change_pct, :volume, :market_cap, :ts)",
            rows,
        )
        await db.commit()


async def upsert_ohlcv(rows: list[dict]) -> None:
    if not rows:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            "INSERT OR REPLACE INTO ohlcv_daily "
            "(symbol, date, open, high, low, close, volume) "
            "VALUES (:symbol, :date, :open, :high, :low, :close, :volume)",
            rows,
        )
        await db.commit()


async def insert_history(rows: list[dict]) -> None:
    """Upsert long historical OHLCV rows into ohlcv_history (idempotent)."""
    if _mysql_store():
        return await asyncio.to_thread(_ms().insert_history, rows)
    if not rows:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            "INSERT OR REPLACE INTO ohlcv_history "
            "(symbol, date, open, high, low, close, adj_close, volume) "
            "VALUES (:symbol, :date, :open, :high, :low, :close, :adj_close, :volume)",
            rows,
        )
        await db.commit()


async def insert_insight(row: dict) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO ai_insights "
            "(symbol, insight_type, content, sentiment, confidence, generated_at) "
            "VALUES (:symbol, :insight_type, :content, :sentiment, :confidence, :generated_at)",
            row,
        )
        await db.commit()


async def insert_news(rows: list[dict]) -> None:
    if not rows:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            "INSERT OR IGNORE INTO news_cache "
            "(title, source, url, summary, published_at, cached_at) "
            "VALUES (:title, :source, :url, :summary, :published_at, :cached_at)",
            rows,
        )
        await db.commit()


async def insert_prediction(row: dict) -> int:
    if _mysql_store():
        return await asyncio.to_thread(_ms().insert_prediction, row)
    row = {**{k: None for k in _PREDICTION_MIGRATIONS}, **row}  # tolerate missing band fields
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "INSERT INTO predictions "
            "(symbol, model, horizon_days, direction, prob_up, meta_prob, act, confidence, "
            " kelly_frac, sentiment, last_close, pred_return, pred_price, conf_low, conf_high, "
            " regime, generated_at, target_date, iv_atm, iv_skew, conf_low_90, conf_high_90) "
            "VALUES (:symbol,:model,:horizon_days,:direction,:prob_up,:meta_prob,:act,"
            ":confidence,:kelly_frac,:sentiment,:last_close,:pred_return,:pred_price,"
            ":conf_low,:conf_high,:regime,:generated_at,:target_date,:iv_atm,:iv_skew,"
            ":conf_low_90,:conf_high_90)",
            row,
        )
        await db.commit()
        return cur.lastrowid


async def insert_outcome(row: dict) -> None:
    """Record a resolved prediction's realized outcome (idempotent on prediction_id)."""
    if _mysql_store():
        return await asyncio.to_thread(_ms().insert_outcome, row)
    row = {"in_band": None, **row}
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO prediction_outcomes "
            "(prediction_id, actual_return, correct, pnl_after_costs, resolved_at, in_band) "
            "VALUES (:prediction_id,:actual_return,:correct,:pnl_after_costs,:resolved_at,:in_band)",
            row,
        )
        await db.commit()


async def get_due_predictions(as_of_date: str) -> list[dict]:
    """Predictions whose target_date has passed and that have no outcome yet."""
    if _mysql_store():
        return await asyncio.to_thread(_ms().get_due_predictions, as_of_date)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT p.* FROM predictions p "
            "LEFT JOIN prediction_outcomes o ON p.id = o.prediction_id "
            "WHERE o.prediction_id IS NULL AND p.target_date IS NOT NULL "
            "AND p.target_date <= ? ORDER BY p.target_date ASC",
            (as_of_date,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def get_latest_predictions(symbol: str | None = None, limit: int = 50) -> list[dict]:
    """
    Most recent prediction per symbol (newest first). With `symbol`, returns that symbol's
    recent predictions; without, one latest row per symbol (leaderboard source).
    """
    if _mysql_store():
        return await asyncio.to_thread(_ms().get_latest_predictions, symbol, limit)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if symbol:
            async with db.execute(
                "SELECT * FROM predictions WHERE symbol=? ORDER BY generated_at DESC LIMIT ?",
                (symbol.upper(), limit),
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]
        # One latest row per symbol via a max(generated_at) self-join.
        async with db.execute(
            "SELECT p.* FROM predictions p JOIN ("
            "  SELECT symbol, MAX(generated_at) AS g FROM predictions GROUP BY symbol"
            ") m ON p.symbol=m.symbol AND p.generated_at=m.g ORDER BY p.confidence DESC LIMIT ?",
            (limit,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def get_prediction_history(symbol: str, limit: int = 100) -> list[dict]:
    """A symbol's predictions left-joined with their resolved outcomes (newest first)."""
    if _mysql_store():
        return await asyncio.to_thread(_ms().get_prediction_history, symbol, limit)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT p.*, o.actual_return, o.correct, o.pnl_after_costs, o.resolved_at, o.in_band "
            "FROM predictions p LEFT JOIN prediction_outcomes o ON p.id=o.prediction_id "
            "WHERE p.symbol=? ORDER BY p.generated_at DESC LIMIT ?",
            (symbol.upper(), limit),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def get_resolved_outcomes() -> list[dict]:
    """Resolved predictions with their stated confidence + correctness (live-calibration source)."""
    if _mysql_store():
        return await asyncio.to_thread(_ms().get_resolved_outcomes)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT p.confidence, p.meta_prob, o.correct, o.actual_return, o.pnl_after_costs "
            "FROM prediction_outcomes o JOIN predictions p ON p.id=o.prediction_id",
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def get_calibration_buckets(model: str = "live") -> list[dict]:
    """Realized hit-rate per confidence bucket (for the reliability strip)."""
    if _mysql_store():
        return await asyncio.to_thread(_ms().get_calibration_buckets, model)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM calibration_buckets WHERE model=? ORDER BY bucket ASC",
            (model,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def upsert_calibration(rows: list[dict]) -> None:
    if _mysql_store():
        return await asyncio.to_thread(_ms().upsert_calibration, rows)
    if not rows:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            "INSERT OR REPLACE INTO calibration_buckets "
            "(model, bucket, stated_conf, realized_hit, n, updated_at) "
            "VALUES (:model,:bucket,:stated_conf,:realized_hit,:n,:updated_at)",
            rows,
        )
        await db.commit()


async def insert_sentiment(rows: list[dict]) -> None:
    """Upsert FinBERT sentiment scores (idempotent on url)."""
    if not rows:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            "INSERT OR REPLACE INTO news_sentiment "
            "(url, symbol, source, title, score, label, published_at, scored_at) "
            "VALUES (:url, :symbol, :source, :title, :score, :label, :published_at, :scored_at)",
            rows,
        )
        await db.commit()


async def get_unscored_news(limit: int = 200) -> list[dict]:
    """news_cache rows not yet present in news_sentiment."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT n.title, n.url, n.summary, n.published_at FROM news_cache n "
            "LEFT JOIN news_sentiment s ON n.url = s.url "
            "WHERE s.url IS NULL ORDER BY n.cached_at DESC LIMIT ?",
            (limit,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def get_symbol_sentiment(symbol: str, days: int = 7) -> list[dict]:
    """Recent sentiment rows for a symbol (and general-market rows)."""
    import time

    cutoff = int((time.time() - days * 86400) * 1000)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM news_sentiment WHERE (symbol=? OR symbol='') AND scored_at>=? "
            "ORDER BY scored_at DESC",
            (symbol.upper(), cutoff),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def insert_options_iv(rows: list[dict]) -> None:
    """Upsert daily options IV/skew snapshots (idempotent on symbol+date)."""
    if not rows:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            "INSERT OR REPLACE INTO options_iv "
            "(symbol, date, spot, atm_iv, skew, term_slope, n_contracts, snapshot_at) "
            "VALUES (:symbol, :date, :spot, :atm_iv, :skew, :term_slope, :n_contracts, :snapshot_at)",
            rows,
        )
        await db.commit()


async def get_latest_options_iv(symbol: str) -> dict | None:
    """Most recent options IV/skew snapshot for a symbol (None if none recorded yet)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM options_iv WHERE symbol=? ORDER BY date DESC LIMIT 1",
            (symbol.upper(),),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def log_ingestion(job: str, status: str, rows: int = 0, message: str = "") -> None:
    import time

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO ingestion_log (job, status, rows, message, ts) VALUES (?,?,?,?,?)",
            (job, status, rows, message, int(time.time() * 1000)),
        )
        await db.commit()


# ── Reads ─────────────────────────────────────────────────────────────────────


async def get_latest_snapshots(limit: int = 50) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM price_snapshots ORDER BY ts DESC LIMIT ?", (limit,)
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def get_symbol_history(symbol: str, limit: int = 100) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM price_snapshots WHERE symbol=? ORDER BY ts DESC LIMIT ?",
            (symbol, limit),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def get_ohlcv(symbol: str, days: int = 30) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM ohlcv_daily WHERE symbol=? ORDER BY date DESC LIMIT ?",
            (symbol.upper(), days),
        ) as cur:
            rows = [dict(r) for r in await cur.fetchall()]
            return list(reversed(rows))


async def get_history(symbol: str, start: str | None = None) -> list[dict]:
    """
    Return the full historical OHLCV series for a symbol, oldest→newest.
    Pass start='YYYY-MM-DD' to limit to rows on/after that date.
    """
    if _mysql_store():
        return await asyncio.to_thread(_ms().get_history, symbol, start)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if start:
            sql = "SELECT * FROM ohlcv_history WHERE symbol=? AND date>=? ORDER BY date ASC"
            params: tuple = (symbol.upper(), start)
        else:
            sql = "SELECT * FROM ohlcv_history WHERE symbol=? ORDER BY date ASC"
            params = (symbol.upper(),)
        async with db.execute(sql, params) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def history_summary() -> list[dict]:
    """Per-symbol row count + date range — used to verify a backfill."""
    if _mysql_store():
        return await asyncio.to_thread(_ms().history_summary)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT symbol, COUNT(*) AS rows, MIN(date) AS first, MAX(date) AS last "
            "FROM ohlcv_history GROUP BY symbol ORDER BY symbol"
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def get_latest_insights(limit: int = 20, symbol: str | None = None) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if symbol:
            async with db.execute(
                "SELECT * FROM ai_insights WHERE symbol=? ORDER BY generated_at DESC LIMIT ?",
                (symbol.upper(), limit),
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]
        async with db.execute(
            "SELECT * FROM ai_insights ORDER BY generated_at DESC LIMIT ?", (limit,)
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def get_latest_verdict(symbol: str) -> dict | None:
    """Most recent verifier verdict (insight_type='prediction') logged for one symbol."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM ai_insights WHERE symbol=? AND insight_type='prediction' "
            "ORDER BY generated_at DESC LIMIT 1",
            (symbol.upper(),),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def get_latest_verdicts(limit: int = 50) -> list[dict]:
    """Most recent verifier verdict per symbol (leaderboard badge source)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT i.* FROM ai_insights i JOIN ("
            "  SELECT symbol, MAX(generated_at) AS g FROM ai_insights "
            "  WHERE insight_type='prediction' GROUP BY symbol"
            ") m ON i.symbol=m.symbol AND i.generated_at=m.g "
            "WHERE i.insight_type='prediction' ORDER BY i.generated_at DESC LIMIT ?",
            (limit,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def get_recent_news(limit: int = 20) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM news_cache ORDER BY cached_at DESC LIMIT ?", (limit,)
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def get_ingestion_log(limit: int = 20) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM ingestion_log ORDER BY ts DESC LIMIT ?", (limit,)
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def prune_old_snapshots(keep_days: int = 7) -> int:
    """Delete price snapshots older than keep_days to bound DB size."""
    import time

    cutoff = int((time.time() - keep_days * 86400) * 1000)
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("DELETE FROM price_snapshots WHERE ts < ?", (cutoff,))
        await db.commit()
        return cur.rowcount


async def prune_ingestion_log(keep_days: int = 30) -> int:
    """Delete ingestion_log rows older than keep_days (0 keeps everything)."""
    if keep_days <= 0:
        return 0
    import time

    cutoff = int((time.time() - keep_days * 86400) * 1000)
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("DELETE FROM ingestion_log WHERE ts < ?", (cutoff,))
        await db.commit()
        return cur.rowcount
