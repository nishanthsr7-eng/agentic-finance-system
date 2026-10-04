"""
FLUX — Payments write API (/db/*)
=================================

Write endpoints backing the Payments page. The /db/* read endpoints in
user_api.py are the source of truth for rendering; these mutate the same
MySQL tables so a payment, recurring bill, protocol toggle, reward claim or
source-account switch survives reloads and logout/login (unlike the old
localStorage-only behaviour).

Mounted by main.py via `app.include_router(payments_router)`. Handlers are
plain `def` so FastAPI runs the blocking PyMySQL driver in its threadpool.
user_id is resolved from the bearer token (see backend/auth.py).

Routes
------
  POST /db/transactions        — record a payment / request / split (validated)
  POST /db/recurring           — add a recurring bill
  POST /db/security/toggle     — flip a protocol/security setting by position
  POST /db/rewards/claim       — claim a reward by reward_key
  POST /db/accounts/activate   — switch the primary (active) source account
"""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from . import mysql_db as M
from .auth import require_user

log = logging.getLogger("flux.payments")

payments_router = APIRouter(prefix="/db", tags=["payments"])

# Monthly discretionary spend cap used for the "Monthly Headroom" gauge when an
# account carries no explicit credit_limit. Kept here so it is a real,
# single-sourced number rather than a magic constant scattered in the UI.
DEFAULT_MONTHLY_LIMIT = 100_000.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _active_account(user_id: int, name: str | None = None) -> dict | None:
    if name:
        row = M.query_one("SELECT * FROM accounts WHERE user_id=%s AND name=%s", (user_id, name))
        if row:
            return row
    return M.query_one(
        "SELECT * FROM accounts WHERE user_id=%s ORDER BY active DESC, sort_order LIMIT 1",
        (user_id,),
    )


def _available(acct: dict) -> float:
    """Spendable headroom on an account = its running balance (this is what we
    debit/credit). For a credit line with no tracked balance we fall back to the
    credit limit. Mirrors the Payments page so client and server agree."""
    bal = float(acct.get("balance") or 0)
    if bal > 0:
        return bal
    if (acct.get("acct_type") or "").lower() == "credit":
        return float(acct.get("credit_limit") or 0)
    return bal


# ── Transactions ─────────────────────────────────────────────────────────────


class TxnRequest(BaseModel):
    title: str = Field(..., max_length=200)
    amount: float = Field(..., description="Signed: negative = outgoing, positive = incoming")
    category: str = Field("transfer", max_length=40)
    account: str | None = Field(None, max_length=120)


def _map_txn(row: dict) -> dict:
    """Return a transaction in the legacy localStorage shape the pages read."""
    return {
        "id": row.get("ext_id") or ("tx_" + str(row.get("id"))),
        "title": row.get("title"),
        "date": (
            row["tx_date"].isoformat()
            if isinstance(row.get("tx_date"), datetime)
            else str(row.get("tx_date"))
        ),
        "amount": float(row.get("amount") or 0),
        "category": row.get("category"),
        "account": row.get("account"),
    }


@payments_router.post("/transactions")
def create_transaction(body: TxnRequest, user_id: int = Depends(require_user)) -> dict:
    """Record a transaction, debiting/crediting the source account atomically.

    Outgoing amounts are validated against the account's available headroom so
    a transfer can never overdraw the source."""
    amount = float(body.amount)
    if amount == 0:
        raise HTTPException(400, "Amount must be non-zero")

    title = body.title.strip()
    if not title:
        raise HTTPException(400, "Title is required")

    acct = _active_account(user_id, body.account)
    acct_name = acct["name"] if acct else (body.account or "")
    outgoing = amount < 0

    if outgoing and acct:
        avail = _available(acct)
        if abs(amount) > avail + 0.005:
            raise HTTPException(
                400,
                f"Insufficient funds: ₹{abs(amount):,.0f} exceeds available "
                f"₹{avail:,.0f} on {acct_name}",
            )

    tx_type = "income" if amount > 0 else ("transfer" if body.category == "transfer" else "expense")
    ext_id = "tx_" + secrets.token_hex(8)  # unique even for same-millisecond requests
    now = _now()

    with M.get_conn() as conn, conn.cursor() as cur:
        try:
            conn.begin()
            cur.execute(
                "INSERT INTO transactions (user_id, ext_id, title, tx_date, amount, "
                "category, account, tx_type) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (user_id, ext_id, title, now, amount, body.category, acct_name, tx_type),
            )
            tx_id = cur.lastrowid
            if acct:
                # Debit/credit the source account balance.
                cur.execute(
                    "UPDATE accounts SET balance = balance + %s WHERE id=%s",
                    (amount, acct["id"]),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    row = M.query_one("SELECT * FROM transactions WHERE id=%s", (tx_id,))
    new_bal = None
    if acct:
        refreshed = M.query_one("SELECT balance FROM accounts WHERE id=%s", (acct["id"],))
        new_bal = float(refreshed["balance"]) if refreshed else None
    log.info("txn #%d user=%d %s %.2f (%s)", tx_id, user_id, title, amount, acct_name)
    return {"ok": True, "transaction": _map_txn(row), "account_balance": new_bal}


# ── Recurring ────────────────────────────────────────────────────────────────


class RecurringRequest(BaseModel):
    title: str = Field(..., max_length=160)
    amount: float = Field(..., gt=0)
    due_day: int = Field(..., ge=1, le=31)
    category: str = Field("subscription", max_length=40)


@payments_router.post("/recurring")
def add_recurring(body: RecurringRequest, user_id: int = Depends(require_user)) -> dict:
    title = body.title.strip()
    if not title:
        raise HTTPException(400, "Title is required")
    with M.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO recurring_payments (user_id, title, amount, due_day, category, active) "
            "VALUES (%s,%s,%s,%s,%s,1)",
            (user_id, title, body.amount, body.due_day, body.category),
        )
    rows = M.query("SELECT * FROM recurring_payments WHERE user_id=%s ORDER BY due_day", (user_id,))
    return {"ok": True, "recurring": rows}


# ── Security / protocol toggles ──────────────────────────────────────────────


class ToggleRequest(BaseModel):
    index: int = Field(..., ge=0, description="0-based position among security settings")
    enabled: bool


@payments_router.post("/security/toggle")
def toggle_security(body: ToggleRequest, user_id: int = Depends(require_user)) -> dict:
    """Flip the enabled flag of the Nth security setting (sorted by sort_order).
    Positional so it matches the page's protocol list without leaking keys."""
    rows = M.query(
        "SELECT id FROM security_settings WHERE user_id=%s ORDER BY sort_order", (user_id,)
    )
    if body.index >= len(rows):
        raise HTTPException(404, "No such security setting")
    with M.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE security_settings SET enabled=%s WHERE id=%s",
            (1 if body.enabled else 0, rows[body.index]["id"]),
        )
    settings = M.query(
        "SELECT enabled FROM security_settings WHERE user_id=%s ORDER BY sort_order", (user_id,)
    )
    return {"ok": True, "settings": [bool(s["enabled"]) for s in settings]}


# ── Rewards ──────────────────────────────────────────────────────────────────


class ClaimRequest(BaseModel):
    reward_key: str = Field(..., max_length=80)


@payments_router.post("/rewards/claim")
def claim_reward(body: ClaimRequest, user_id: int = Depends(require_user)) -> dict:
    row = M.query_one(
        "SELECT id, claimed FROM rewards WHERE user_id=%s AND reward_key=%s",
        (user_id, body.reward_key),
    )
    if not row:
        raise HTTPException(404, "Reward not found")
    if row["claimed"]:
        raise HTTPException(409, "Reward already claimed")
    with M.get_conn() as conn, conn.cursor() as cur:
        cur.execute("UPDATE rewards SET claimed=1 WHERE id=%s", (row["id"],))
    rows = M.query("SELECT * FROM rewards WHERE user_id=%s", (user_id,))
    return {"ok": True, "rewards": rows}


# ── Account switch ───────────────────────────────────────────────────────────


class ActivateRequest(BaseModel):
    account_id: int | None = None
    name: str | None = Field(None, max_length=120)


@payments_router.post("/accounts/activate")
def activate_account(body: ActivateRequest, user_id: int = Depends(require_user)) -> dict:
    if body.account_id is not None:
        target = M.query_one(
            "SELECT id FROM accounts WHERE user_id=%s AND id=%s", (user_id, body.account_id)
        )
    elif body.name:
        target = M.query_one(
            "SELECT id FROM accounts WHERE user_id=%s AND name=%s", (user_id, body.name)
        )
    else:
        raise HTTPException(400, "Provide account_id or name")
    if not target:
        raise HTTPException(404, "Account not found")

    with M.get_conn() as conn, conn.cursor() as cur:
        try:
            conn.begin()
            cur.execute("UPDATE accounts SET active=0 WHERE user_id=%s", (user_id,))
            cur.execute("UPDATE accounts SET active=1 WHERE id=%s", (target["id"],))
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    rows = M.query(
        "SELECT id, user_id, name, acct_type, balance, credit_limit, active, "
        "card_masked, expiry_masked, sort_order FROM accounts WHERE user_id=%s ORDER BY sort_order",
        (user_id,),
    )
    return {"ok": True, "accounts": rows}
