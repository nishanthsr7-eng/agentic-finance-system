"""
FLUX — MySQL persistence layer
==============================

A second, relational home for FLUX data. SQLite (`flux_market.db`) stays the
live-ingestion store; this MySQL database is the seeded "ultimate" dataset that
every page reads from through the new /db/* API endpoints.

Connection settings come from .env (see config.Settings):
    MYSQL_HOST  MYSQL_PORT  MYSQL_USER  MYSQL_PASSWORD  MYSQL_DB

Driver: PyMySQL (sync). FastAPI runs the /db endpoints as plain `def`, so they
execute in the threadpool and the blocking driver is fine.

Schema groups
-------------
1. Personal finance  — users, accounts, transactions, portfolio(+holdings),
                       recurring_payments, contacts, vault_goals(+deposits),
                       credit_scores(+credit_factors), rewards,
                       security_settings, devices, security_events
2. Market data       — asset_catalog, price_snapshots, ohlcv_daily,
                       ohlcv_history, ai_insights, news_cache, predictions,
                       prediction_outcomes, calibration_buckets,
                       news_sentiment, options_iv, ingestion_log
3. Site content      — faqs, job_openings, team_members
"""

from __future__ import annotations

import logging
import queue
import time
from contextlib import contextmanager

import certifi
import pymysql
from pymysql.constants import SERVER_STATUS
from pymysql.cursors import DictCursor

from .config import settings

log = logging.getLogger("flux.mysql")


# ── Connection ──────────────────────────────────────────────────────────────


def _conn_kwargs(include_db: bool = True) -> dict:
    kw = dict(
        host=settings.MYSQL_HOST,
        port=settings.MYSQL_PORT,
        user=settings.MYSQL_USER,
        password=settings.MYSQL_PASSWORD,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=True,
    )
    if include_db:
        kw["database"] = settings.MYSQL_DB
    if settings.MYSQL_SSL:
        # Managed tiers (TiDB Serverless, Aiven, PlanetScale) refuse plaintext.
        # Passing ssl_ca is what actually flips PyMySQL into TLS — a bare
        # ssl={} dict is falsy there and would silently connect unencrypted.
        # certifi ships the CA bundle those providers' certs chain to, and is
        # already present via httpx, so it works identically on Windows and in
        # the deployment container.
        kw["ssl_ca"] = settings.MYSQL_SSL_CA or certifi.where()
        kw["ssl_verify_cert"] = True
        kw["ssl_verify_identity"] = True
    return kw


# Idle connections as (conn, last_used_monotonic). LIFO so the warmest one is
# reused first and the rest age out. Thread-safe: /db routes run in the
# threadpool and the scheduler uses executors.
_pool: queue.LifoQueue = queue.LifoQueue()


def _checkout():
    while True:
        try:
            conn, last_used = _pool.get_nowait()
        except queue.Empty:
            return pymysql.connect(**_conn_kwargs(True))
        try:
            if time.monotonic() - last_used > settings.MYSQL_POOL_PING_AFTER_S:
                conn.ping(reconnect=True)
            return conn
        except Exception:  # noqa: BLE001 — dead connection: drop it, try the next
            _close_quietly(conn)


def _checkin(conn) -> None:
    # Never pool a connection left mid-transaction (a caller that began and
    # neither committed nor rolled back) or beyond the size cap.
    in_txn = conn.server_status & SERVER_STATUS.SERVER_STATUS_IN_TRANS
    if in_txn or not conn.open or _pool.qsize() >= settings.MYSQL_POOL_SIZE:
        _close_quietly(conn)
        return
    _pool.put((conn, time.monotonic()))


def _close_quietly(conn) -> None:
    try:
        conn.close()
    except Exception:  # noqa: BLE001 — already closed / broken
        pass


def close_pool() -> None:
    """Close every idle pooled connection (shutdown, tests)."""
    while True:
        try:
            conn, _ = _pool.get_nowait()
        except queue.Empty:
            return
        _close_quietly(conn)


@contextmanager
def get_conn(include_db: bool = True):
    """Yield a PyMySQL connection.

    With MYSQL_POOL_SIZE > 0 it is borrowed from a small pool and returned on
    a clean exit; on an exception it is closed instead, so a half-used
    connection is never handed to the next caller. Without a pool (the
    default) or for the database-less admin connection, it is opened and
    closed per call as before.
    """
    if settings.MYSQL_POOL_SIZE <= 0 or not include_db:
        conn = pymysql.connect(**_conn_kwargs(include_db))
        try:
            yield conn
        finally:
            conn.close()
        return

    conn = _checkout()
    try:
        yield conn
    except BaseException:
        _close_quietly(conn)
        raise
    _checkin(conn)


def query(sql: str, params: tuple | None = None) -> list[dict]:
    """Run a SELECT and return rows as dicts."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchall()


def query_one(sql: str, params: tuple | None = None) -> dict | None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchone()


def ping() -> bool:
    """True if the MySQL database is reachable (used by /health)."""
    try:
        with get_conn() as conn:
            conn.ping(reconnect=True)
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("MySQL ping failed: %s", e)
        return False


# ── Schema ──────────────────────────────────────────────────────────────────
# Baseline only (applied by migrations/001_baseline.py). Don't edit tables
# here any more: add a new file to backend/migrations/ instead.
# One CREATE TABLE per entry; executed in order so foreign keys resolve.

DDL: list[str] = [
    # ---- 1. Personal finance --------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS users (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        name        VARCHAR(120) NOT NULL,
        email       VARCHAR(190) UNIQUE,
        flux_id     VARCHAR(60)  UNIQUE,
        phone       VARCHAR(40),
        avatar      VARCHAR(255),
        plan        VARCHAR(40)  DEFAULT 'FLUX Black',
        currency    VARCHAR(8)   DEFAULT 'INR',
        city        VARCHAR(80),
        created_at  DATETIME     DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS accounts (
        id           INT AUTO_INCREMENT PRIMARY KEY,
        user_id      INT NOT NULL,
        name         VARCHAR(120) NOT NULL,
        acct_type    VARCHAR(40) DEFAULT 'credit',      -- credit | savings | wallet
        balance      DECIMAL(16,2) DEFAULT 0,
        card_masked  VARCHAR(40),
        card_real    VARCHAR(40),
        expiry_masked VARCHAR(10),
        expiry_real  VARCHAR(10),
        active       TINYINT(1) DEFAULT 0,
        sort_order   INT DEFAULT 0,
        CONSTRAINT fk_acct_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS transactions (
        id        INT AUTO_INCREMENT PRIMARY KEY,
        user_id   INT NOT NULL,
        ext_id    VARCHAR(40),                          -- legacy 'tx_0001' id
        title     VARCHAR(200) NOT NULL,
        tx_date   DATETIME NOT NULL,
        amount    DECIMAL(16,2) NOT NULL,               -- +income / -expense
        category  VARCHAR(40),                          -- income|investment|essentials|...
        account   VARCHAR(120),
        tx_type   VARCHAR(20),                          -- income | expense | transfer
        CONSTRAINT fk_tx_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        INDEX idx_tx_user_date (user_id, tx_date DESC),
        INDEX idx_tx_category (category)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolio (
        user_id    INT PRIMARY KEY,
        equity_pct INT DEFAULT 45,
        crypto_pct INT DEFAULT 30,
        cash_pct   INT DEFAULT 25,
        goal       DECIMAL(16,2) DEFAULT 15000000,
        CONSTRAINT fk_pf_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolio_holdings (
        id         INT AUTO_INCREMENT PRIMARY KEY,
        user_id    INT NOT NULL,
        symbol     VARCHAR(40) NOT NULL,
        name       VARCHAR(120),
        asset_type VARCHAR(20),                          -- equity | crypto | etf
        quantity   DECIMAL(20,8) NOT NULL,
        avg_price  DECIMAL(16,4) NOT NULL,
        currency   VARCHAR(8) DEFAULT 'INR',
        CONSTRAINT fk_hold_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        INDEX idx_hold_user (user_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS recurring_payments (
        id        INT AUTO_INCREMENT PRIMARY KEY,
        user_id   INT NOT NULL,
        title     VARCHAR(160) NOT NULL,
        amount    DECIMAL(16,2) NOT NULL,
        due_day   INT,
        category  VARCHAR(40),
        active     TINYINT(1) DEFAULT 1,
        CONSTRAINT fk_rec_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS contacts (
        id        INT AUTO_INCREMENT PRIMARY KEY,
        user_id   INT NOT NULL,
        name      VARCHAR(120) NOT NULL,
        flux_id   VARCHAR(80),
        initial   VARCHAR(4),
        favorite  TINYINT(1) DEFAULT 0,
        CONSTRAINT fk_con_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS vault_goals (
        id        INT AUTO_INCREMENT PRIMARY KEY,
        user_id   INT NOT NULL,
        name      VARCHAR(160) NOT NULL,
        icon      VARCHAR(16),
        target    DECIMAL(16,2) NOT NULL,
        saved     DECIMAL(16,2) DEFAULT 0,
        monthly   DECIMAL(16,2) DEFAULT 0,
        deadline  DATE,
        CONSTRAINT fk_vg_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS vault_deposits (
        id         INT AUTO_INCREMENT PRIMARY KEY,
        user_id    INT NOT NULL,
        goal_id    INT,
        goal_name  VARCHAR(160),
        amount     DECIMAL(16,2) NOT NULL,
        dep_date   DATE NOT NULL,
        note       VARCHAR(200),
        CONSTRAINT fk_vd_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        CONSTRAINT fk_vd_goal FOREIGN KEY (goal_id) REFERENCES vault_goals(id) ON DELETE SET NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS credit_scores (
        user_id     INT PRIMARY KEY,
        score       INT NOT NULL,
        max_score   INT DEFAULT 900,
        rating      VARCHAR(20),                          -- Excellent | Good | Fair | Poor
        bureau      VARCHAR(40) DEFAULT 'CIBIL',
        updated_at  DATE,
        CONSTRAINT fk_cs_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS credit_factors (
        id        INT AUTO_INCREMENT PRIMARY KEY,
        user_id   INT NOT NULL,
        factor    VARCHAR(80) NOT NULL,                  -- Payment History, Credit Utilization, ...
        status    VARCHAR(20),                           -- Excellent | Good | Fair | Poor
        value_txt VARCHAR(60),                           -- '100% on-time', '24%', ...
        weight_pct INT,
        impact    VARCHAR(20),                           -- High | Medium | Low
        sort_order INT DEFAULT 0,
        CONSTRAINT fk_cf_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS credit_history (
        id        INT AUTO_INCREMENT PRIMARY KEY,
        user_id   INT NOT NULL,
        month     CHAR(7) NOT NULL,                      -- YYYY-MM
        score     INT NOT NULL,
        CONSTRAINT fk_ch_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        UNIQUE KEY uq_ch (user_id, month)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS rewards (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        user_id     INT NOT NULL,
        reward_key  VARCHAR(80) NOT NULL,
        title       VARCHAR(160),
        points      INT DEFAULT 0,
        claimed     TINYINT(1) DEFAULT 0,
        CONSTRAINT fk_rw_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS security_settings (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        user_id     INT NOT NULL,
        setting_key VARCHAR(80) NOT NULL,                -- 2fa, biometric, txn_alerts, ...
        label       VARCHAR(160),
        enabled     TINYINT(1) DEFAULT 0,
        sort_order  INT DEFAULT 0,
        CONSTRAINT fk_ss_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS devices (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        user_id     INT NOT NULL,
        device_name VARCHAR(120),
        os          VARCHAR(60),
        browser     VARCHAR(60),
        location    VARCHAR(120),
        last_active DATETIME,
        trusted     TINYINT(1) DEFAULT 1,
        current_session TINYINT(1) DEFAULT 0,
        CONSTRAINT fk_dev_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS security_events (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        user_id     INT NOT NULL,
        event_type  VARCHAR(60),                         -- login | password_change | alert ...
        description VARCHAR(255),
        location    VARCHAR(120),
        severity    VARCHAR(20) DEFAULT 'info',          -- info | warning | critical
        event_at    DATETIME,
        CONSTRAINT fk_se_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        INDEX idx_se_user (user_id, event_at DESC)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS trades (
        id         INT AUTO_INCREMENT PRIMARY KEY,
        user_id    INT NOT NULL,
        symbol     VARCHAR(40) NOT NULL,                  -- marketplace ticker (AAPL, NVDA, …)
        name       VARCHAR(160),
        side       VARCHAR(4) NOT NULL,                   -- BUY | SELL
        quantity   DECIMAL(20,6) NOT NULL,
        price      DECIMAL(16,4) NOT NULL,                -- per-share execution price
        amount     DECIMAL(18,2) NOT NULL,                -- quantity * price
        currency   VARCHAR(8) DEFAULT 'USD',
        trade_date DATETIME NOT NULL,
        CONSTRAINT fk_trade_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        INDEX idx_trade_user_date (user_id, trade_date DESC),
        INDEX idx_trade_symbol (symbol)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    # ---- 2. Market data -------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS asset_catalog (
        symbol     VARCHAR(40) PRIMARY KEY,
        name       VARCHAR(160),
        sub        VARCHAR(40),                           -- short ticker
        category   VARCHAR(20),                           -- crypto|stocks|forex|indices|commodities
        sector     VARCHAR(60),
        icon       VARCHAR(255)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS price_snapshots (
        id          BIGINT AUTO_INCREMENT PRIMARY KEY,
        symbol      VARCHAR(40) NOT NULL,
        asset_type  VARCHAR(20) NOT NULL,
        name        VARCHAR(160),
        price       DOUBLE NOT NULL,
        change_pct  DOUBLE,
        volume      DOUBLE,
        market_cap  DOUBLE,
        ts          BIGINT NOT NULL,
        INDEX idx_ps_symbol_ts (symbol, ts DESC),
        INDEX idx_ps_ts (ts DESC)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS ohlcv_daily (
        symbol  VARCHAR(40) NOT NULL,
        date    CHAR(10) NOT NULL,
        open    DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
        volume  BIGINT,
        PRIMARY KEY (symbol, date),
        INDEX idx_ohlcv_date (date DESC)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS ohlcv_history (
        symbol    VARCHAR(40) NOT NULL,
        date      CHAR(10) NOT NULL,
        open      DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
        adj_close DOUBLE,
        volume    BIGINT,
        PRIMARY KEY (symbol, date),
        INDEX idx_hist_sym_date (symbol, date DESC)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_insights (
        id           BIGINT AUTO_INCREMENT PRIMARY KEY,
        symbol       VARCHAR(40) NOT NULL,
        insight_type VARCHAR(40) NOT NULL,
        content      TEXT NOT NULL,
        sentiment    VARCHAR(20),
        confidence   INT,
        generated_at BIGINT NOT NULL,
        INDEX idx_ins_ts (generated_at DESC),
        INDEX idx_ins_symbol (symbol, generated_at DESC)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS news_cache (
        id           BIGINT AUTO_INCREMENT PRIMARY KEY,
        title        VARCHAR(400),
        source       VARCHAR(120),
        url          VARCHAR(500) UNIQUE,
        summary      TEXT,
        published_at VARCHAR(40),
        cached_at    BIGINT NOT NULL,
        INDEX idx_news_cached (cached_at DESC)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS predictions (
        id           BIGINT AUTO_INCREMENT PRIMARY KEY,
        symbol       VARCHAR(40) NOT NULL,
        model        VARCHAR(60) NOT NULL,
        horizon_days INT NOT NULL,
        direction    VARCHAR(10),
        prob_up      DOUBLE, meta_prob DOUBLE,
        act          INT, confidence INT,
        kelly_frac   DOUBLE, sentiment DOUBLE, last_close DOUBLE,
        pred_return  DOUBLE, pred_price DOUBLE,
        conf_low     DOUBLE, conf_high DOUBLE,
        regime       VARCHAR(20),
        iv_atm       DOUBLE, iv_skew DOUBLE,
        generated_at BIGINT NOT NULL,
        target_date  CHAR(10),
        INDEX idx_pred_sym (symbol, generated_at DESC)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS prediction_outcomes (
        prediction_id   BIGINT PRIMARY KEY,
        actual_return   DOUBLE,
        correct         INT,
        pnl_after_costs DOUBLE,
        resolved_at     BIGINT
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS calibration_buckets (
        model        VARCHAR(60) NOT NULL,
        bucket       INT NOT NULL,
        stated_conf  DOUBLE,
        realized_hit DOUBLE,
        n            INT,
        updated_at   BIGINT,
        PRIMARY KEY (model, bucket)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS news_sentiment (
        url          VARCHAR(500) PRIMARY KEY,
        symbol       VARCHAR(40),
        source       VARCHAR(40),
        title        VARCHAR(400),
        score        DOUBLE,
        label        VARCHAR(20),
        published_at VARCHAR(40),
        scored_at    BIGINT NOT NULL,
        INDEX idx_sent_symbol (symbol, scored_at DESC)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS options_iv (
        symbol      VARCHAR(40) NOT NULL,
        date        CHAR(10) NOT NULL,
        spot        DOUBLE, atm_iv DOUBLE, skew DOUBLE, term_slope DOUBLE,
        n_contracts INT,
        snapshot_at BIGINT NOT NULL,
        PRIMARY KEY (symbol, date),
        INDEX idx_opt_symbol (symbol, date DESC)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS ingestion_log (
        id      BIGINT AUTO_INCREMENT PRIMARY KEY,
        job     VARCHAR(60) NOT NULL,
        status  VARCHAR(10) NOT NULL,
        `rows`  INT DEFAULT 0,
        message VARCHAR(400),
        ts      BIGINT NOT NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    # ---- 3. Site content ------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS faqs (
        id        INT AUTO_INCREMENT PRIMARY KEY,
        category  VARCHAR(60),
        question  VARCHAR(400) NOT NULL,
        answer    TEXT NOT NULL,
        sort_order INT DEFAULT 0
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS job_openings (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        title       VARCHAR(160) NOT NULL,
        department  VARCHAR(80),
        location    VARCHAR(120),
        emp_type    VARCHAR(40),                          -- Full-time | Contract | Intern
        description TEXT,
        active      TINYINT(1) DEFAULT 1
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS team_members (
        id        INT AUTO_INCREMENT PRIMARY KEY,
        name      VARCHAR(120) NOT NULL,
        role      VARCHAR(120),
        bio       TEXT,
        avatar    VARCHAR(255),
        sort_order INT DEFAULT 0
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
]


def create_database() -> None:
    """Create the FLUX database if it does not yet exist (connects without a db)."""
    with get_conn(include_db=False) as conn, conn.cursor() as cur:
        cur.execute(
            f"CREATE DATABASE IF NOT EXISTS `{settings.MYSQL_DB}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
    log.info("MySQL database `%s` ready", settings.MYSQL_DB)


def init_schema() -> None:
    """Create the database and every table (idempotent)."""
    create_database()
    with get_conn() as conn, conn.cursor() as cur:
        for stmt in DDL:
            cur.execute(stmt)
    log.info("MySQL schema initialised (%d tables)", len(DDL))
    from .migrate import run_migrations

    run_migrations()
