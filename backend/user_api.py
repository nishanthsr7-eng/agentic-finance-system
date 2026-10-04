"""
FLUX — MySQL-backed API (/db/*)
===============================

Read endpoints serving the seeded MySQL dataset to every page. Mounted by
main.py via `app.include_router(db_router)`. Handlers are plain `def` so
FastAPI runs the blocking PyMySQL driver in its threadpool.

Personal-finance endpoints resolve user_id from the Authorization bearer token
(see backend/auth.py). The old ?user_id= query override is gone — clients can
no longer read another user's data by changing a URL parameter. With
AUTH_REQUIRED=false, unauthenticated requests act as the demo user (id=1).

Routes
------
  GET /db/health                 — MySQL reachability + per-table row counts
  GET /db/user                   — profile
  GET /db/accounts               — bank/cards
  GET /db/transactions           — ledger (?limit, ?category)
  GET /db/portfolio              — allocation + holdings
  GET /db/recurring              — recurring payments
  GET /db/contacts               — payee contacts
  GET /db/rewards                — rewards / streaks
  GET /db/security               — settings + devices + events
  GET /db/bootstrap              — all of the above user data in one call (page-load hydration)
  GET /db/faqs                   — FAQ entries (?category)
  GET /db/careers                — open job postings
  GET /db/team                   — team members
  GET /db/market/catalog         — asset catalogue (?category)
  GET /db/market/snapshots       — latest price snapshots (?limit)
  GET /db/market/insights        — AI insights (?symbol, ?limit)
  GET /db/market/news            — cached news (?limit)
  GET /db/market/predictions     — latest prediction per symbol (leaderboard)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from . import mysql_db as M
from .auth import require_user

db_router = APIRouter(prefix="/db", tags=["mysql"])


# ── meta ──────────────────────────────────────────────────────────────────────


@db_router.get("/health")
def db_health() -> dict:
    if not M.ping():
        return {"ok": False, "error": "MySQL unreachable"}
    tables = [
        "users",
        "accounts",
        "transactions",
        "portfolio_holdings",
        "recurring_payments",
        "contacts",
        "rewards",
        "devices",
        "security_events",
        "faqs",
        "job_openings",
        "team_members",
        "asset_catalog",
        "price_snapshots",
        "ohlcv_daily",
        "ohlcv_history",
        "ai_insights",
        "news_cache",
        "predictions",
    ]
    counts = {}
    for t in tables:
        try:
            counts[t] = M.query_one(f"SELECT COUNT(*) AS n FROM `{t}`")["n"]
        except Exception:  # noqa: BLE001
            counts[t] = None
    return {"ok": True, "database": M.settings.MYSQL_DB, "counts": counts}


# ── personal finance ────────────────────────────────────────────────────────


@db_router.get("/user")
def get_user(user_id: int = Depends(require_user)) -> dict:
    row = M.query_one("SELECT * FROM users WHERE id=%s", (user_id,))
    return {"user": row}


class _Conn:
    """query/query_one over one open connection, so a request that needs
    several result sets pays for a single TiDB connect + TLS handshake."""

    def __init__(self, conn):
        self._conn = conn

    def query(self, sql: str, params: tuple | None = None) -> list[dict]:
        with self._conn.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.fetchall()

    def query_one(self, sql: str, params: tuple | None = None) -> dict | None:
        with self._conn.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.fetchone()


def _accounts(db, user_id: int) -> dict:
    # Never ship unmasked card numbers / expiry dates to the client.
    # credit_limit comes from migrations/005_accounts_credit_limit.py.
    rows = db.query(
        "SELECT id, user_id, name, acct_type, balance, "
        "COALESCE(credit_limit, 0) AS credit_limit, active, card_masked, "
        "expiry_masked, sort_order FROM accounts WHERE user_id=%s ORDER BY sort_order",
        (user_id,),
    )
    return {"accounts": rows}


def _transactions(db, user_id: int, limit: int, category: str | None = None) -> dict:
    if category:
        rows = db.query(
            "SELECT * FROM transactions WHERE user_id=%s AND category=%s "
            "ORDER BY tx_date DESC LIMIT %s",
            (user_id, category, limit),
        )
    else:
        rows = db.query(
            "SELECT * FROM transactions WHERE user_id=%s ORDER BY tx_date DESC LIMIT %s",
            (user_id, limit),
        )
    return {"transactions": rows, "count": len(rows)}


def _portfolio(db, user_id: int) -> dict:
    alloc = db.query_one("SELECT * FROM portfolio WHERE user_id=%s", (user_id,))
    holdings = db.query("SELECT * FROM portfolio_holdings WHERE user_id=%s", (user_id,))
    return {"allocation": alloc, "holdings": holdings}


def _recurring(db, user_id: int) -> dict:
    rows = db.query(
        "SELECT * FROM recurring_payments WHERE user_id=%s ORDER BY due_day", (user_id,)
    )
    return {"recurring": rows}


def _contacts(db, user_id: int) -> dict:
    rows = db.query(
        "SELECT * FROM contacts WHERE user_id=%s ORDER BY favorite DESC, name", (user_id,)
    )
    return {"contacts": rows}


def _rewards(db, user_id: int) -> dict:
    rows = db.query("SELECT * FROM rewards WHERE user_id=%s", (user_id,))
    return {"rewards": rows}


def _security(db, user_id: int) -> dict:
    settings = db.query(
        "SELECT * FROM security_settings WHERE user_id=%s ORDER BY sort_order", (user_id,)
    )
    devices = db.query(
        "SELECT * FROM devices WHERE user_id=%s ORDER BY last_active DESC", (user_id,)
    )
    events = db.query(
        "SELECT * FROM security_events WHERE user_id=%s ORDER BY event_at DESC", (user_id,)
    )
    return {"settings": settings, "devices": devices, "events": events}


@db_router.get("/accounts")
def get_accounts(user_id: int = Depends(require_user)) -> dict:
    return _accounts(M, user_id)


@db_router.get("/transactions")
def get_transactions(
    user_id: int = Depends(require_user),
    limit: int = Query(500, le=2000),
    category: str | None = None,
) -> dict:
    return _transactions(M, user_id, limit, category)


@db_router.get("/trades")
def get_trades(
    user_id: int = Depends(require_user),
    limit: int = Query(200, le=1000),
    side: str | None = None,
) -> dict:
    """Paper-trade history (stocks and crypto), newest first. Optional
    ?side=BUY|SELL. (Used to JOIN asset_catalog on category='stocks', but the
    catalogue holds no stocks, so every trade was filtered out.)"""
    params: list = [user_id]
    sql = "SELECT t.* FROM trades t WHERE t.user_id=%s "
    if side and side.upper() in ("BUY", "SELL"):
        sql += "AND t.side=%s "
        params.append(side.upper())
    sql += "ORDER BY t.trade_date DESC LIMIT %s"
    params.append(limit)
    rows = M.query(sql, tuple(params))
    return {"trades": rows, "count": len(rows)}


@db_router.get("/portfolio")
def get_portfolio(user_id: int = Depends(require_user)) -> dict:
    return _portfolio(M, user_id)


@db_router.get("/recurring")
def get_recurring(user_id: int = Depends(require_user)) -> dict:
    return _recurring(M, user_id)


@db_router.get("/contacts")
def get_contacts(user_id: int = Depends(require_user)) -> dict:
    return _contacts(M, user_id)


@db_router.get("/rewards")
def get_rewards(user_id: int = Depends(require_user)) -> dict:
    return _rewards(M, user_id)


@db_router.get("/security")
def get_security(user_id: int = Depends(require_user)) -> dict:
    return _security(M, user_id)


@db_router.get("/bootstrap")
def get_bootstrap(
    user_id: int = Depends(require_user),
    tx_limit: int = Query(2000, le=2000),
) -> dict:
    """Everything the app hydrates on page load, in one request over one
    connection (replaces 7 parallel /db/* calls, ~10 TiDB connects)."""
    with M.get_conn() as conn:
        db = _Conn(conn)
        return {
            "transactions": _transactions(db, user_id, tx_limit),
            "accounts": _accounts(db, user_id),
            "portfolio": _portfolio(db, user_id),
            "recurring": _recurring(db, user_id),
            "contacts": _contacts(db, user_id),
            "security": _security(db, user_id),
            "rewards": _rewards(db, user_id),
        }


# ── site content ──────────────────────────────────────────────────────────────


@db_router.get("/faqs")
def get_faqs(category: str | None = None) -> dict:
    if category:
        rows = M.query("SELECT * FROM faqs WHERE category=%s ORDER BY sort_order", (category,))
    else:
        rows = M.query("SELECT * FROM faqs ORDER BY category, sort_order")
    return {"faqs": rows}


@db_router.get("/careers")
def get_careers() -> dict:
    rows = M.query("SELECT * FROM job_openings WHERE active=1 ORDER BY department, title")
    return {"jobs": rows}


@db_router.get("/team")
def get_team() -> dict:
    rows = M.query("SELECT * FROM team_members ORDER BY sort_order")
    return {"team": rows}


# ── market (from MySQL) ─────────────────────────────────────────────────────


@db_router.get("/market/catalog")
def market_catalog(category: str | None = None) -> dict:
    if category:
        rows = M.query("SELECT * FROM asset_catalog WHERE category=%s ORDER BY symbol", (category,))
    else:
        rows = M.query("SELECT * FROM asset_catalog ORDER BY category, symbol")
    return {"assets": rows}


@db_router.get("/market/snapshots")
def market_snapshots(limit: int = Query(50, le=500)) -> dict:
    rows = M.query("SELECT * FROM price_snapshots ORDER BY ts DESC LIMIT %s", (limit,))
    return {"snapshots": rows}


@db_router.get("/market/insights")
def market_insights(symbol: str | None = None, limit: int = Query(20, le=200)) -> dict:
    if symbol:
        rows = M.query(
            "SELECT * FROM ai_insights WHERE symbol=%s ORDER BY generated_at DESC LIMIT %s",
            (symbol.upper(), limit),
        )
    else:
        rows = M.query("SELECT * FROM ai_insights ORDER BY generated_at DESC LIMIT %s", (limit,))
    return {"insights": rows}


@db_router.get("/market/news")
def market_news(limit: int = Query(20, le=200)) -> dict:
    rows = M.query("SELECT * FROM news_cache ORDER BY cached_at DESC LIMIT %s", (limit,))
    return {"news": rows}


@db_router.get("/market/predictions")
def market_predictions(limit: int = Query(50, le=200)) -> dict:
    rows = M.query(
        "SELECT p.* FROM predictions p JOIN ("
        "  SELECT symbol, MAX(generated_at) AS g FROM predictions GROUP BY symbol"
        ") m ON p.symbol=m.symbol AND p.generated_at=m.g "
        "ORDER BY p.confidence DESC LIMIT %s",
        (limit,),
    )
    return {"predictions": rows}
