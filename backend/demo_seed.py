"""
FLUX — Rolling demo-user seed (user 1)
======================================

Rebuilds the demo account so it always looks like a live one, whatever the
date: a year of transactions ending today, future goal deadlines, a current
credit report month, recent devices and logins, and a short paper-trade book
priced from closes ingestion has already stored.

  seed_demo_user(today)   — wipe and rewrite user 1 in ONE MySQL transaction
  ensure_demo_fresh()     — reseed only if it hasn't run today (startup self-heal)

Deterministic: every past month draws from an RNG seeded with that month, so
history stays put from one day to the next and only the recent rows change.
The same date always produces the same rows. Other users are never touched.

Balances add up:
  savings / wallet  = opening balance + that account's transactions
  credit card       = credit limit + this month's card spend (earlier
                      statements are treated as paid; no settlement rows,
                      because the dashboard would count them as spending)

No network calls and no pandas: trade prices come from the local
ohlcv_daily / ohlcv_history tables (SQLite) that the OHLCV jobs fill.
"""

from __future__ import annotations

import logging
import random
import sqlite3
from datetime import date, datetime, timedelta, timezone

log = logging.getLogger("flux.demo_seed")

DEMO_USER_ID = 1
SEED_CASH_USD = 100_000.0  # matches trading_api.SEED_CASH_USD
STATE_KEY = "demo_seeded_on"

CC, SAV, WAL = "HDFC Platinum", "ICICI Wealth", "FLUX Wallet"
CC_LIMIT = 250_000.0
OPENING = {SAV: 60_000.0, WAL: 110_000.0}

# Every table that holds demo-user rows this module owns (children first).
USER_TABLES = (
    "vault_deposits",
    "vault_goals",
    "credit_factors",
    "credit_history",
    "credit_scores",
    "portfolio_holdings",
    "portfolio",
    "recurring_payments",
    "contacts",
    "transactions",
    "trades",
    "accounts",
    "rewards",
    "security_settings",
    "devices",
    "security_events",
)

# Paper-trade symbols: in the marketplace stock quote pool (so they can be
# sold back) and in ingestion's OHLCV_SYMBOLS (so there are stored closes).
TRADE_SYMBOLS = {
    "AAPL": "Apple Inc.",
    "NVDA": "NVIDIA Corp.",
    "TSLA": "Tesla Inc.",
    "MSFT": "Microsoft Corp.",
}

GROCERY = ["BigBasket", "Zepto", "Blinkit", "DMart Ready", "Nature's Basket"]
FOOD = [
    "Swiggy",
    "Zomato",
    "Blue Tokai Coffee",
    "Third Wave Coffee",
    "Truffles",
    "Meghana Foods",
    "Domino's",
    "McDonald's",
    "Subway",
    "CTR Malleshwaram",
]
RIDE = ["Uber", "Ola", "Rapido", "Namma Metro Recharge"]
SHOP = ["Amazon.in", "Flipkart", "Myntra", "Decathlon", "Croma", "IKEA Bengaluru", "Uniqlo"]
FUN = ["BookMyShow", "PVR Cinemas", "Steam", "PlayStation Store"]
HEALTH = ["Apollo Pharmacy", "1mg", "Practo Consult", "Cult.fit Day Pass"]
FRIENDS = ["Alex", "Sanjay", "Meera", "Priya", "Rohan"]


# ── date helpers ─────────────────────────────────────────────────────────────

# The demo user lives in Bengaluru and the browser shows naive tx_date values as
# local time, so the story clock is IST. 00:05 UTC is 05:35 IST, same date.
IST = timezone(timedelta(hours=5, minutes=30))


def ist_now() -> datetime:
    return datetime.now(IST).replace(tzinfo=None, microsecond=0)


def _add_months(d: date, n: int) -> date:
    total = d.year * 12 + (d.month - 1) + n
    y, m = divmod(total, 12)
    return date(y, m + 1, 1)


def _days_in_month(y: int, m: int) -> int:
    return (_add_months(date(y, m, 1), 1) - timedelta(days=1)).day


def _next_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _ts(rng: random.Random, d: date, h_lo: int = 8, h_hi: int = 23) -> datetime:
    """Human-looking time of day, evenings weighted heavier than mornings."""
    hours = range(h_lo, h_hi + 1)
    h = rng.choices(hours, weights=[2 if x < 12 else 3 if x < 18 else 4 for x in hours])[0]
    return datetime(d.year, d.month, d.day, h, rng.randint(0, 59))


def _fmt(dt: datetime | date) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S") if isinstance(dt, datetime) else dt.isoformat()


# ── transactions ─────────────────────────────────────────────────────────────


def _month_transactions(y: int, m: int, start: date, today: date) -> list[tuple]:
    """One month of believable spending. (tx_datetime, title, amount, category, account)."""
    rng = random.Random(y * 100 + m)
    last = _days_in_month(y, m)
    out: list[tuple] = []

    def on(day: int) -> date | None:
        d = date(y, m, min(day, last))
        return d if start <= d <= today else None

    def add(d: date | None, when: datetime | None, title, amount, cat, acct):
        if d is not None:
            out.append((when or _ts(rng, d), title, round(amount, 2), cat, acct))

    # Draw every random number whether or not the day is in range, so a month's
    # rows don't depend on how much of it has happened yet.
    pay_day = _next_weekday(date(y, m, 1))
    add(on(pay_day.day), _ts(rng, pay_day, 9, 11), "Salary — Acme Corp", 188000, "income", SAV)

    if rng.random() < 0.4:
        day = rng.randint(12, 24)
        amt = rng.choice([15000, 22500, 28000, 36500])
        add(on(day), None, "Freelance — UI audit (Upwork)", amt, "income", SAV)

    fixed = [
        (22, "Rent — Koramangala 5th Block", -22000, "essentials", SAV),
        (3, "Airtel Xstream Fiber", -1499, "essentials", CC),
        (18, "Jio Postpaid", -599, "essentials", CC),
        (7, "BESCOM Electricity", -rng.randint(1750, 3400), "essentials", CC),
        (8, "BWSSB Water", -rng.randint(380, 560), "essentials", SAV),
        (10, "Netflix India", -649, "subscription", CC),
        (9, "Spotify Premium", -149, "subscription", WAL),
        (4, "ChatGPT Plus", -1800, "subscription", CC),
        (12, "iCloud+ 200GB", -75, "subscription", CC),
        (15, "YouTube Premium", -149, "subscription", WAL),
        (5, "AWS Cloud Services", -rng.randint(7800, 9600), "business", CC),
        (6, "Cult.fit Membership", -1500, "lifestyle", CC),
        (25, "Zerodha — ETF SIP", -10000, "investment", SAV),
        (2, "Vanguard S&P 500 SIP", -45000, "investment", SAV),
    ]
    for day, title, amt, cat, acct in fixed:
        lo, hi = (6, 21) if cat in ("subscription", "business") else (8, 23)
        when = _ts(rng, date(y, m, min(day, last)), lo, hi)
        add(on(day), when, title, amt, cat, acct)

    def burst(n_lo, n_hi, day_lo, day_hi, h_lo, h_hi, pick):
        for _ in range(rng.randint(n_lo, n_hi)):
            day = rng.randint(day_lo, day_hi)
            when = _ts(rng, date(y, m, min(day, last)), h_lo, h_hi)
            title, amt, cat, acct = pick()
            add(on(day), when, title, amt, cat, acct)

    burst(
        4,
        6,
        1,
        28,
        10,
        21,
        lambda: (f"{rng.choice(GROCERY)} / Groceries", -rng.randint(420, 4600), "essentials", CC),
    )
    burst(
        9,
        14,
        1,
        28,
        9,
        23,
        lambda: (
            rng.choice(FOOD),
            -rng.choice([183, 219, 268, 312, 345, 428, 516, 624, 745, 890, 1145]),
            "lifestyle",
            WAL if rng.random() < 0.5 else CC,
        ),
    )
    burst(
        6,
        10,
        1,
        28,
        8,
        22,
        lambda: (
            rng.choice(RIDE),
            -rng.choice([89, 124, 156, 189, 234, 312, 418, 540]),
            "lifestyle",
            WAL,
        ),
    )
    burst(
        2,
        2,
        5,
        20,
        8,
        20,
        lambda: ("Indian Oil — COCO Indiranagar", -rng.randint(1700, 2300), "essentials", CC),
    )
    burst(
        2,
        4,
        2,
        27,
        11,
        22,
        lambda: (
            rng.choice(SHOP),
            -rng.choice([549, 899, 1249, 1899, 2450, 3299, 4799, 7490]),
            "lifestyle",
            CC,
        ),
    )
    burst(
        1,
        2,
        3,
        27,
        17,
        23,
        lambda: (rng.choice(FUN), -rng.choice([350, 520, 760, 999, 1450]), "lifestyle", CC),
    )
    burst(
        1,
        2,
        2,
        27,
        9,
        20,
        lambda: (rng.choice(HEALTH), -rng.choice([260, 385, 540, 820, 1240]), "essentials", WAL),
    )

    def split():
        f = rng.choice(FRIENDS)
        if rng.random() < 0.45:
            return (
                f"UPI from {f} — split settle",
                rng.choice([420, 650, 845, 1250, 1600]),
                "transfer",
                WAL,
            )
        return (
            f"UPI to {f} — dinner split",
            -rng.choice([380, 560, 720, 980, 1340]),
            "transfer",
            WAL,
        )

    burst(2, 4, 1, 28, 10, 23, split)

    if rng.random() < 0.35:
        day = rng.randint(8, 24)
        when = _ts(rng, date(y, m, min(day, last)), 19, 23)
        title = rng.choice(["WazirX — BTC buy", "CoinDCX — ETH buy", "CoinDCX — SOL buy"])
        add(on(day), when, title, -rng.choice([5000, 8000, 12000, 15000]), "investment", SAV)
    return out


def _today_transactions(today: date) -> list[tuple]:
    """A normal morning-to-evening on today's date, so the 'Today' panels are alive."""
    rng = random.Random(today.toordinal())
    picks = [
        (
            7,
            10,
            rng.choice(["Blue Tokai Coffee", "Third Wave Coffee"]),
            -rng.choice([240, 268, 312]),
            WAL,
        ),
        (8, 11, rng.choice(RIDE), -rng.choice([124, 156, 184, 234]), WAL),
        (12, 14, rng.choice(["Swiggy", "Zomato"]), -rng.choice([326, 428, 516]), CC),
        (17, 20, rng.choice(GROCERY) + " / Groceries", -rng.randint(420, 1800), CC),
        (0, 1, "Swiggy Instamart", -rng.choice([183, 219, 268]), WAL),  # before the 05:35 reseed
    ]
    out = []
    for h_lo, h_hi, title, amt, acct in picks:
        when = datetime(
            today.year, today.month, today.day, rng.randint(h_lo, h_hi), rng.randint(0, 59)
        )
        cat = "essentials" if "Groceries" in title or "Instamart" in title else "lifestyle"
        out.append((when, title, float(amt), cat, acct))
    return out


def _fill_quiet_days(rows: list[tuple], today: date) -> list[tuple]:
    """One small spend on any of the last 30 days that has none, so the
    Analysis cashflow heatmap shows activity every day. Seeded by the day
    itself, so a given date always gets the same row on every reseed."""
    spent = {r[0].date() for r in rows if r[2] < 0}
    out = []
    for back in range(1, 30):
        day = today - timedelta(days=back)
        if day in spent:
            continue
        rng = random.Random(day.toordinal() * 13 + 5)
        title, lo, hi, cat = rng.choice(
            [
                (rng.choice(["Blue Tokai Coffee", "Third Wave Coffee"]), 180, 420, "lifestyle"),
                (rng.choice(GROCERY) + " / Groceries", 350, 1600, "essentials"),
                (rng.choice(RIDE), 110, 380, "lifestyle"),
                ("Swiggy Instamart", 160, 540, "essentials"),
            ]
        )
        when = datetime(day.year, day.month, day.day, rng.randint(9, 21), rng.randint(0, 59))
        out.append((when, title, -float(rng.randint(lo, hi)), cat, WAL))
    return out


def build_transactions(today: date, now: datetime) -> list[tuple]:
    start = today - timedelta(days=365)
    rows: list[tuple] = []
    month = date(start.year, start.month, 1)
    while month <= today:
        rows += _month_transactions(month.year, month.month, start, today)
        month = _add_months(month, 1)
    rows += _today_transactions(today)
    rows += _fill_quiet_days(rows, today)
    # Nothing in the future: today's rows only up to the moment of seeding.
    rows = [r for r in rows if r[0] <= now]
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


# ── the whole dataset ────────────────────────────────────────────────────────


def _trades(today: date, now: datetime, closes: dict[str, dict[str, float]]) -> list[tuple]:
    """8-15 paper trades over the last 30 days at that day's stored close.
    (symbol, name, side, qty, price, amount, trade_datetime)"""
    rng = random.Random(today.toordinal() * 7 + 1)
    planned, seen = [], set()
    for _ in range(rng.randint(8, 15)):
        day = _next_weekday(today - timedelta(days=rng.randint(1, 29)))
        if day > today:
            day = today - timedelta(days=3)
        sym = rng.choice(list(TRADE_SYMBOLS))
        # One trade per symbol per day: two identical same-day rows read as a double fill.
        if (sym, day) in seen:
            continue
        seen.add((sym, day))
        when = datetime(day.year, day.month, day.day, rng.randint(14, 19), rng.randint(0, 59))
        planned.append((when, sym, rng.randint(1, 8), rng.random()))
    planned.sort()

    out, held = [], dict.fromkeys(TRADE_SYMBOLS, 0)
    for when, sym, qty, coin in planned:
        if when > now:
            continue
        series = closes.get(sym) or {}
        # That day's close, or the last close before it (weekend / holiday).
        on_or_before = [d for d in series if d <= when.date().isoformat()]
        if not on_or_before:
            continue
        price = round(series[max(on_or_before)], 2)
        if held[sym] > 1 and coin < 0.35:
            side, qty = "SELL", min(qty, held[sym] - 1)
            held[sym] -= qty
        else:
            side = "BUY"
            held[sym] += qty
        out.append((sym, TRADE_SYMBOLS[sym], side, qty, price, round(qty * price, 2), when))
    return out


def build_demo_rows(today: date, now: datetime, closes: dict[str, dict[str, float]]) -> dict:
    """Every row the demo user gets, as plain tuples (no DB access)."""
    txs = build_transactions(today, now)
    month_start = datetime(today.year, today.month, 1)

    bal = dict(OPENING)
    cc_cycle = 0.0
    for when, _t, amt, _c, acct in txs:
        if acct == CC:
            if when >= month_start:
                cc_cycle += amt
        else:
            bal[acct] += amt
    balances = {SAV: round(bal[SAV], 2), WAL: round(bal[WAL], 2), CC: round(CC_LIMIT + cc_cycle, 2)}

    accounts = [
        (
            CC,
            "credit",
            balances[CC],
            "•••• •••• •••• 8842",
            "4532 9982 1104 8842",
            "••/••",
            "08/29",
            1,
            0,
            CC_LIMIT,
        ),
        (
            SAV,
            "savings",
            balances[SAV],
            "•••• •••• •••• 1290",
            "5104 2293 8810 1290",
            "••/••",
            "03/28",
            0,
            1,
            0,
        ),
        (
            WAL,
            "wallet",
            balances[WAL],
            "•••• •••• •••• 4471",
            "6011 5566 7788 4471",
            "••/••",
            "12/27",
            0,
            2,
            0,
        ),
    ]

    transactions = [
        (
            f"demo_{when:%Y%m%d}_{i:04d}",
            title,
            _fmt(when),
            amt,
            cat,
            acct,
            "income"
            if amt > 0 and cat != "transfer"
            else ("transfer" if cat == "transfer" else "expense"),
        )
        for i, (when, title, amt, cat, acct) in enumerate(txs)
    ]

    holdings = [
        ("SPY", "SPDR S&P 500 ETF", "etf", 6, 48900.00, "USD"),
        ("RELIANCE", "Reliance Industries", "equity", 60, 1290.00, "INR"),
        ("TCS", "Tata Consultancy", "equity", 30, 3420.00, "INR"),
        ("INFY", "Infosys Ltd", "equity", 80, 1510.00, "INR"),
        ("BTC", "Bitcoin", "crypto", 0.03, 5400000, "INR"),
        ("ETH", "Ethereum", "crypto", 0.4, 280000, "INR"),
        ("SOL", "Solana", "crypto", 6, 14500, "INR"),
    ]

    # Allocation is derived from the holdings (at cost) and the cash accounts, so the
    # Dashboard's equity / crypto / cash split agrees with the Analysis holdings.
    cost = {"equity": 0.0, "crypto": 0.0}
    for _s, _n, kind, qty, avg, _c in holdings:
        cost["crypto" if kind == "crypto" else "equity"] += qty * avg
    cash_total = balances[SAV] + balances[WAL]
    whole = cost["equity"] + cost["crypto"] + cash_total
    crypto_pct = round(cost["crypto"] / whole * 100)
    cash_pct = round(cash_total / whole * 100)
    portfolio = (100 - crypto_pct - cash_pct, crypto_pct, cash_pct, 15000000)

    recurring = [
        ("AWS Infrastructure", 8500, 5, "business"),
        ("ChatGPT Plus", 1800, 4, "subscription"),
        ("Netflix India", 649, 10, "subscription"),
        ("Spotify Premium", 149, 9, "subscription"),
        ("Rent — Koramangala", 22000, 22, "essentials"),
        ("Zerodha — ETF SIP", 10000, 25, "investment"),
        ("Vanguard S&P 500 SIP", 45000, 2, "investment"),
    ]

    contacts = [
        ("Alex", "alex@flux", "A", 1),
        ("Sanjay", "sanjay@flux", "S", 1),
        ("Meera", "meera@flux", "M", 0),
        ("Priya", "priya@flux", "P", 0),
        ("Rohan", "rohan@flux", "R", 0),
    ]

    # Goals: name, icon, target, saved, monthly, deadline (always in the future)
    goals = [
        ("Emergency Fund", "🛡️", 300000, 87500, 8000, _add_months(today, 4)),
        ("Europe Trip", "✈️", 150000, 42000, 5000, _add_months(today, 7)),
        ("New MacBook", "📱", 180000, 162000, 15000, _add_months(today, 2)),
        ("Down Payment", "🏠", 2500000, 320000, 25000, _add_months(today, 48)),
    ]
    first = date(today.year, today.month, 1)
    goal_cycle = ["Emergency Fund", "Europe Trip", "New MacBook", "Down Payment"]
    deposits = []
    for k in range(6):  # 1st of each of the last 6 months
        d = _add_months(first, -k)
        name = goal_cycle[k % 4]
        amount = next(g[4] for g in goals if g[0] == name)
        deposits.append(
            (name, amount, d, "Monthly auto-deposit" if k % 2 == 0 else "Monthly contribution")
        )

    score_trend = [712, 718, 724, 729, 733, 738, 742, 747, 751, 755, 759, 762]
    credit_history = [(f"{_add_months(first, i - 11):%Y-%m}", s) for i, s in enumerate(score_trend)]
    credit_factors = [
        ("Payment History", "Excellent", "100% on-time", 35, "High", 0),
        ("Credit Utilization", "Good", "24%", 30, "High", 1),
        ("Credit Age", "Good", "6 yr 4 mo", 15, "Medium", 2),
        ("Account Mix", "Excellent", "4 types", 10, "Low", 3),
        ("Hard Inquiries", "Fair", "3 in 12 mo", 10, "Medium", 4),
    ]

    rewards = [
        ("first_payment", "First Payment Sent", 250, 1),
        ("vault_starter", "Opened a Vault Goal", 150, 1),
        ("streak_30", "30-Day Activity Streak", 500, 1),
        ("invest_5l", "₹5L Invested Milestone", 1000, 0),
        ("referral_3", "Referred 3 Friends", 750, 0),
        ("credit_750", "Crossed 750 Credit Score", 600, 1),
    ]

    security_settings = [
        ("2fa", "Two-Factor Authentication", 1, 0),
        ("biometric", "Biometric Unlock", 1, 1),
        ("txn_alerts", "Transaction Alerts", 1, 2),
        ("login_alerts", "New Login Alerts", 1, 3),
        ("intl_block", "Block International Cards", 0, 4),
        ("spend_limit", "Daily Spend Limit", 0, 5),
    ]

    def ago(days: int, hours: int = 0) -> str:
        return _fmt((now - timedelta(days=days, hours=hours)).replace(second=0, microsecond=0))

    devices = [
        (
            'MacBook Pro 14"',
            "macOS 15",
            "Chrome",
            "Bengaluru, IN",
            _fmt(now.replace(microsecond=0)),
            1,
            1,
        ),
        ("iPhone 15 Pro", "iOS 18", "Safari", "Bengaluru, IN", ago(1, 3), 1, 0),
        ("Windows Desktop", "Windows 11", "Edge", "Bengaluru, IN", ago(4, 2), 1, 0),
        ("iPad Air", "iPadOS 18", "Safari", "Mumbai, IN", ago(13, 5), 0, 0),
    ]
    big = max(
        (t for t in txs if t[2] <= -20000 and t[0] >= now - timedelta(days=14)),
        key=lambda t: t[0],
        default=None,
    )
    events = [
        ("login", "Successful login", "Bengaluru, IN", "info", _fmt(now.replace(microsecond=0))),
        ("device", "New device added: iPad Air", "Mumbai, IN", "info", ago(13, 5)),
        ("password_change", "Password changed", "Bengaluru, IN", "info", ago(9, 6)),
        ("login", "Blocked login attempt", "Unknown, RU", "critical", ago(6, 14)),
    ]
    if big:
        events.append(
            (
                "alert",
                f"Large transaction flagged ₹{abs(big[2]):,.0f}",
                "Bengaluru, IN",
                "warning",
                _fmt(big[0]),
            )
        )

    trades = _trades(today, now, closes)
    cash = SEED_CASH_USD
    for _s, _n, side, _q, _p, amount, _w in trades:
        cash += -amount if side == "BUY" else amount

    return {
        "accounts": accounts,
        "balances": balances,
        "transactions": transactions,
        "holdings": holdings,
        "portfolio": portfolio,
        "recurring": recurring,
        "contacts": contacts,
        "goals": goals,
        "deposits": deposits,
        "credit_history": credit_history,
        "credit_factors": credit_factors,
        "credit_as_of": first,
        "rewards": rewards,
        "security_settings": security_settings,
        "devices": devices,
        "events": events,
        "trades": trades,
        "wallet_cash": round(cash, 2),
    }


# ── stored closes (SQLite, or TiDB with MARKET_STORE=mysql; no network) ─────────────────────────────────


def load_closes(start: date, end: date) -> dict[str, dict[str, float]]:
    """{symbol: {YYYY-MM-DD: close}} from ohlcv_history and ohlcv_daily."""
    from .db import DB_PATH, _mysql_store

    out: dict[str, dict[str, float]] = {}
    syms = list(TRADE_SYMBOLS)
    lo, hi = (start - timedelta(days=7)).isoformat(), end.isoformat()
    tables = ("ohlcv_history", "ohlcv_daily")
    if _mysql_store():  # history lives in TiDB
        tables = ("ohlcv_daily",)
        try:
            from .market_store_mysql import get_closes

            for r in get_closes(syms, lo, hi):
                if r["close"]:
                    out.setdefault(r["symbol"], {})[r["date"]] = float(r["close"])
        except Exception as exc:
            log.warning("demo seed: reading TiDB closes failed: %s", exc)
    if not DB_PATH.exists():
        return out
    marks = ",".join("?" * len(syms))
    try:
        with sqlite3.connect(DB_PATH) as db:
            for table in tables:  # daily wins on overlap
                for sym, d, close in db.execute(
                    f"SELECT symbol, date, close FROM {table} "
                    f"WHERE symbol IN ({marks}) AND date BETWEEN ? AND ?",
                    (*syms, lo, hi),
                ):
                    if close:
                        out.setdefault(sym, {})[d] = float(close)
    except sqlite3.Error as exc:
        log.warning("demo seed: reading stored closes failed: %s", exc)
    return out


# ── writing ──────────────────────────────────────────────────────────────────


def write_demo_rows(conn, data: dict, today: date) -> None:
    """Replace user 1's rows with `data` inside one transaction."""
    uid = DEMO_USER_ID
    with conn.cursor() as cur:
        try:
            conn.begin()
            for t in USER_TABLES:
                cur.execute(f"DELETE FROM `{t}` WHERE user_id=%s", (uid,))

            cur.executemany(
                "INSERT INTO accounts (user_id,name,acct_type,balance,card_masked,card_real,"
                "expiry_masked,expiry_real,active,sort_order,credit_limit) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                [(uid, *a) for a in data["accounts"]],
            )
            cur.executemany(
                "INSERT INTO transactions (user_id,ext_id,title,tx_date,amount,category,account,tx_type) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                [(uid, *t) for t in data["transactions"]],
            )
            cur.execute(
                "INSERT INTO portfolio (user_id,equity_pct,crypto_pct,cash_pct,goal) VALUES (%s,%s,%s,%s,%s)",
                (uid, *data["portfolio"]),
            )
            cur.executemany(
                "INSERT INTO portfolio_holdings (user_id,symbol,name,asset_type,quantity,avg_price,currency) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                [(uid, *h) for h in data["holdings"]],
            )
            cur.executemany(
                "INSERT INTO recurring_payments (user_id,title,amount,due_day,category,active) "
                "VALUES (%s,%s,%s,%s,%s,1)",
                [(uid, *r) for r in data["recurring"]],
            )
            cur.executemany(
                "INSERT INTO contacts (user_id,name,flux_id,initial,favorite) VALUES (%s,%s,%s,%s,%s)",
                [(uid, *c) for c in data["contacts"]],
            )
            goal_ids = {}
            for name, icon, target, saved, monthly, deadline in data["goals"]:
                cur.execute(
                    "INSERT INTO vault_goals (user_id,name,icon,target,saved,monthly,deadline) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (uid, name, icon, target, saved, monthly, deadline.isoformat()),
                )
                goal_ids[name] = cur.lastrowid
            cur.executemany(
                "INSERT INTO vault_deposits (user_id,goal_id,goal_name,amount,dep_date,note) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                [
                    (uid, goal_ids[n], n, a, d.isoformat(), note)
                    for n, a, d, note in data["deposits"]
                ],
            )
            cur.execute(
                "INSERT INTO credit_scores (user_id,score,max_score,rating,bureau,updated_at) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (
                    uid,
                    data["credit_history"][-1][1],
                    900,
                    "Excellent",
                    "CIBIL",
                    data["credit_as_of"].isoformat(),
                ),
            )
            cur.executemany(
                "INSERT INTO credit_factors (user_id,factor,status,value_txt,weight_pct,impact,sort_order) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                [(uid, *f) for f in data["credit_factors"]],
            )
            cur.executemany(
                "INSERT INTO credit_history (user_id,month,score) VALUES (%s,%s,%s)",
                [(uid, *h) for h in data["credit_history"]],
            )
            cur.executemany(
                "INSERT INTO rewards (user_id,reward_key,title,points,claimed) VALUES (%s,%s,%s,%s,%s)",
                [(uid, *r) for r in data["rewards"]],
            )
            cur.executemany(
                "INSERT INTO security_settings (user_id,setting_key,label,enabled,sort_order) "
                "VALUES (%s,%s,%s,%s,%s)",
                [(uid, *s) for s in data["security_settings"]],
            )
            cur.executemany(
                "INSERT INTO devices (user_id,device_name,os,browser,location,last_active,trusted,current_session) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                [(uid, *d) for d in data["devices"]],
            )
            cur.executemany(
                "INSERT INTO security_events (user_id,event_type,description,location,severity,event_at) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                [(uid, *e) for e in data["events"]],
            )
            if data["trades"]:
                cur.executemany(
                    "INSERT INTO trades (user_id,symbol,name,side,quantity,price,amount,currency,"
                    "asset_type,trade_date) VALUES (%s,%s,%s,%s,%s,%s,%s,'USD','stocks',%s)",
                    [
                        (uid, s, n, side, q, p, a, _fmt(w))
                        for s, n, side, q, p, a, w in data["trades"]
                    ],
                )
            cur.execute(
                "INSERT INTO trading_wallet (user_id, cash_usd, seeded_usd) VALUES (%s,%s,%s) "
                "ON DUPLICATE KEY UPDATE cash_usd=VALUES(cash_usd), seeded_usd=VALUES(seeded_usd), "
                "updated_at=CURRENT_TIMESTAMP",
                (uid, data["wallet_cash"], SEED_CASH_USD),
            )
            cur.execute(
                "INSERT INTO app_state (k, v) VALUES (%s,%s) "
                "ON DUPLICATE KEY UPDATE v=VALUES(v), updated_at=CURRENT_TIMESTAMP",
                (STATE_KEY, today.isoformat()),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def seed_demo_user(today: date | None = None, now: datetime | None = None) -> dict:
    """Rebuild the demo user for `today`. Blocking; run it in an executor."""
    from . import mysql_db as M

    now = now or ist_now()
    today = today or now.date()
    data = build_demo_rows(today, now, load_closes(today - timedelta(days=30), today))
    with M.get_conn() as conn:  # app_state: migrations/006_app_state.sql
        write_demo_rows(conn, data, today)
    summary = {
        "date": today.isoformat(),
        "transactions": len(data["transactions"]),
        "trades": len(data["trades"]),
    }
    log.info("demo reseed ok: %s", summary)
    return summary


def ensure_demo_fresh(today: date | None = None) -> dict | None:
    """Reseed only if it hasn't happened today (covers days Render slept through 00:05)."""
    from . import mysql_db as M

    today = today or ist_now().date()
    with M.get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT v FROM app_state WHERE k=%s", (STATE_KEY,))
        row = cur.fetchone()
    if row and row["v"] == today.isoformat():
        log.info("demo seed already fresh for %s", today)
        return None
    return seed_demo_user(today)
