"""
FLUX — Authentication
=====================

Self-contained auth using only the Python stdlib (no PyJWT/bcrypt deps):

  * Passwords  — PBKDF2-HMAC-SHA256, 260k iterations, 16-byte random salt.
                 Stored as "pbkdf2$<iterations>$<salt_b64>$<hash_b64>".
  * Tokens     — HMAC-SHA256-signed payload (JWT-like, but no header/alg
                 negotiation surface): base64url(json).base64url(sig).
                 Payload: {"uid": int, "exp": unix_ts}. 24h lifetime.
  * Secret     — settings.AUTH_SECRET if set, else generated once and
                 persisted to <app-data>/flux/auth_secret.key so sessions
                 survive restarts.

Routes
------
  POST /auth/register   {name, email, password}        → {token, user}
  POST /auth/login      {email, password}              → {token, user}
  GET  /auth/me         (Authorization: Bearer <tok>)  → {user}

`require_user` is the FastAPI dependency that protects personal-data routes:
it returns the authenticated user id, or raises 401. When AUTH_REQUIRED=false
(dev convenience) it falls back to the demo user (id=1).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from . import mysql_db as M
from .config import settings
from .ratelimit import rate_limit

log = logging.getLogger("flux.auth")

auth_router = APIRouter(prefix="/auth", tags=["auth"])

TOKEN_TTL_SECONDS = 24 * 3600
PBKDF2_ITERATIONS = 260_000
DEMO_USER_ID = 1
DEMO_PASSWORD = "FluxDemo@123"  # seeded for user 1 only if they have no password yet


# ── Secret resolution ────────────────────────────────────────────────────────


def _secret_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / ".local" / "share")) / "flux"
    base.mkdir(parents=True, exist_ok=True)
    return base / "auth_secret.key"


def _load_secret() -> bytes:
    if getattr(settings, "AUTH_SECRET", ""):
        return settings.AUTH_SECRET.encode()
    p = _secret_path()
    if p.exists():
        return p.read_bytes()
    secret = secrets.token_bytes(32)
    p.write_bytes(secret)
    log.info("Generated new auth secret at %s", p)
    return secret


_SECRET = _load_secret()


# ── Password hashing ─────────────────────────────────────────────────────────


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return "pbkdf2$%d$%s$%s" % (
        PBKDF2_ITERATIONS,
        base64.b64encode(salt).decode(),
        base64.b64encode(dk).decode(),
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        _scheme, iters, salt_b64, hash_b64 = stored.split("$")
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), base64.b64decode(salt_b64), int(iters)
        )
        return hmac.compare_digest(dk, base64.b64decode(hash_b64))
    except Exception:  # noqa: BLE001 — malformed hash → fail closed
        return False


# ── Token mint / verify ──────────────────────────────────────────────────────


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _b64u_dec(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def mint_token(user_id: int) -> str:
    payload = _b64u(
        json.dumps({"uid": user_id, "exp": int(time.time()) + TOKEN_TTL_SECONDS}).encode()
    )
    sig = _b64u(hmac.new(_SECRET, payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{sig}"


def verify_token(token: str) -> int | None:
    """Return the user id if the token is valid and unexpired, else None."""
    try:
        payload_b64, sig_b64 = token.split(".")
        expected = hmac.new(_SECRET, payload_b64.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64u_dec(sig_b64)):
            return None
        payload = json.loads(_b64u_dec(payload_b64))
        if payload.get("exp", 0) < time.time():
            return None
        return int(payload["uid"])
    except Exception:  # noqa: BLE001 — any malformed token → unauthenticated
        return None


# ── FastAPI dependency ───────────────────────────────────────────────────────


def require_user(authorization: str | None = Header(default=None)) -> int:
    """Resolve the authenticated user id from `Authorization: Bearer <token>`.

    With AUTH_REQUIRED=false (dev), unauthenticated requests fall back to the
    demo user instead of failing — flip it on for any real deployment.
    """
    if authorization and authorization.lower().startswith("bearer "):
        uid = verify_token(authorization[7:].strip())
        if uid is not None:
            return uid
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if not settings.AUTH_REQUIRED:
        return DEMO_USER_ID
    raise HTTPException(status_code=401, detail="Authentication required")


def is_admin(token: str | None) -> bool:
    """True only when ADMIN_TOKEN is configured and `token` matches it."""
    expected = settings.ADMIN_TOKEN
    if not expected or not token:
        return False
    return hmac.compare_digest(token, expected)


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    """Gate maintenance routes on the `X-Admin-Token` header. Fails closed:
    with ADMIN_TOKEN unset, every call is refused."""
    if not is_admin(x_admin_token):
        raise HTTPException(status_code=403, detail="Admin token required")


# ── Demo credential seed (called from startup) ──────────────────────


def ensure_auth_schema() -> None:
    """Give the demo user a known password so the seeded dataset stays usable
    after auth goes mandatory. (users.password_hash itself comes from
    backend/migrations/002_users_password_hash.py.)"""
    try:
        demo = M.query_one("SELECT id, password_hash FROM users WHERE id=%s", (DEMO_USER_ID,))
        if demo and not demo.get("password_hash"):
            with M.get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET password_hash=%s WHERE id=%s",
                    (hash_password(DEMO_PASSWORD), DEMO_USER_ID),
                )
            log.warning(
                "Demo user (id=%d) password set to the default — change it via /auth/register flow before exposing this instance.",
                DEMO_USER_ID,
            )
    except Exception as e:  # noqa: BLE001 — MySQL down: routes will surface it
        log.warning("ensure_auth_schema skipped: %s", e)


# ── Routes ───────────────────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


def _public_user(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "password_hash"}


@auth_router.post("/register", dependencies=[Depends(rate_limit("register", 3, 60))])
def register(body: RegisterRequest) -> dict:
    name = body.name.strip()
    email = body.email.strip().lower()
    if not name or len(name) > 120:
        raise HTTPException(400, "Name is required (max 120 chars)")
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        raise HTTPException(400, "Invalid email address")
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if M.query_one("SELECT id FROM users WHERE email=%s", (email,)):
        raise HTTPException(409, "An account with this email already exists")

    with M.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (%s, %s, %s)",
            (name, email, hash_password(body.password)),
        )
        uid = cur.lastrowid
    user = M.query_one("SELECT * FROM users WHERE id=%s", (uid,))
    if user is None:
        raise HTTPException(500, "Account could not be created")
    return {"token": mint_token(uid), "user": _public_user(user)}


@auth_router.post("/login", dependencies=[Depends(rate_limit("login", 10, 60))])
def login(body: LoginRequest) -> dict:
    email = body.email.strip().lower()
    user = M.query_one("SELECT * FROM users WHERE email=%s", (email,))
    # Same error for unknown email and wrong password — don't leak which.
    if (
        not user
        or not user.get("password_hash")
        or not verify_password(body.password, user["password_hash"])
    ):
        raise HTTPException(401, "Invalid email or password")
    return {"token": mint_token(user["id"]), "user": _public_user(user)}


@auth_router.get("/me")
def me(authorization: str | None = Header(default=None)) -> dict:
    uid = require_user(authorization)
    user = M.query_one("SELECT * FROM users WHERE id=%s", (uid,))
    if not user:
        raise HTTPException(404, "User not found")
    return {"user": _public_user(user)}
