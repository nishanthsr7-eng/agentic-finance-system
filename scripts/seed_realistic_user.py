"""
FLUX — Realistic user-data seeder (user 1)
==========================================

Replaces user 1's robotic seed transactions (everything at 10:00, ~15/month,
identical amounts) with a believable 12-month financial life:

  * ~55–70 transactions/month: salary, rent, utilities, subscriptions on their
    real billing days, groceries, food delivery, transport, shopping,
    entertainment, health, occasional travel, SIPs and a monthly card
    settlement that roughly tracks the month's credit-card spend.
  * Human times (mornings/evenings weighted), human amounts (₹183, ₹1,254 —
    not round thousands everywhere), month-to-month variation.
  * Transactions up to *today* so "Today" panels aren't empty.
  * Fresh June trades at current market prices (from price_snapshots) so the
    marketplace trade book doesn't stop on June 1.
  * Account balances retold to match the story.

Deterministic (random.seed) and idempotent: re-running regenerates the same
dataset. Categories stay within the frontend taxonomy:
income / essentials / lifestyle / subscription / investment / business / transfer.
"""

from __future__ import annotations

import random
import sys
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import mysql_db as M  # noqa: E402

random.seed(20260611)

TODAY = date(2026, 6, 11)
START = date(2025, 7, 1)
USER = 1

CC, SAV, WAL = "HDFC Platinum", "ICICI Wealth", "FLUX Wallet"

rows: list[tuple] = []  # (ext_id, title, tx_date, amount, category, account, tx_type)


def ts(d: date, h_lo=8, h_hi=23) -> datetime:
    """Human-looking timestamp — evenings weighted heavier than mornings."""
    h = random.choices(
        range(h_lo, h_hi + 1),
        weights=[2 if x < 12 else 3 if x < 18 else 4 for x in range(h_lo, h_hi + 1)],
    )[0]
    return datetime(d.year, d.month, d.day, h, random.randint(0, 59))


def add(d: datetime, title: str, amount: float, category: str, account: str):
    rows.append(
        (
            f"tx_{uuid.uuid4().hex[:10]}",
            title,
            d,
            round(amount, 2),
            category,
            account,
            "credit" if amount > 0 else "debit",
        )
    )


def next_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def month_days(year: int, month: int):
    d = date(year, month, 1)
    while d.month == month and d <= TODAY:
        yield d
        d += timedelta(days=1)


def clamp(d: date) -> date | None:
    return d if START <= d <= TODAY else None


# ── pools ────────────────────────────────────────────────────────────────────
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
FUN = ["BookMyShow", "PVR Cinemas", "Steam", "PlayStation Store", "Spotify Concert Tix"]
HEALTH = ["Apollo Pharmacy", "1mg", "Practo Consult", "Cult.fit Day Pass"]
FRIENDS = ["Alex", "Sanjay", "Meera", "Priya", "Rohan"]

# one-off travel bursts (month, day, title, amount, category)
TRAVEL = [
    (2025, 10, 17, "IndiGo BLR→GOI", -8456, "lifestyle"),
    (2025, 10, 17, "Zostel Goa — 3 nights", -7200, "lifestyle"),
    (2025, 10, 20, "IndiGo GOI→BLR", -7890, "lifestyle"),
    (2026, 1, 9, "Vistara BLR→DEL (wedding)", -12340, "lifestyle"),
    (2026, 1, 12, "Vistara DEL→BLR", -11875, "lifestyle"),
    (2026, 4, 3, "Shatabdi BLR→MAS", -1245, "lifestyle"),
    (2026, 4, 6, "Shatabdi MAS→BLR", -1245, "lifestyle"),
]

# ── month loop ───────────────────────────────────────────────────────────────
cur = START
while cur <= TODAY:
    y, m = cur.year, cur.month
    cc_spend = 0.0  # tracked → next month's card settlement

    def cc_add(d, title, amt, cat):
        global cc_spend
        cc_spend += -amt
        add(d, title, amt, cat, CC)

    # income — salary credited 1st (next weekday), annual hike in Jan
    sal_day = clamp(next_weekday(date(y, m, 1)))
    if sal_day:
        salary = 188000 if date(y, m, 1) >= date(2026, 1, 1) else 182000
        add(ts(sal_day, 9, 11), "Salary — Acme Corp", salary, "income", SAV)
    # freelance income, some months
    if random.random() < 0.4:
        d = clamp(date(y, m, random.randint(12, 24)))
        if d:
            add(
                ts(d),
                "Freelance — UI audit (Upwork)",
                random.choice([15000, 22500, 28000, 36500]),
                "income",
                SAV,
            )

    # fixed bills (matching recurring_payments due days)
    fixed = [
        (22, "Rent — Koramangala 5th Block", -22000, "essentials", SAV),
        (3, "Airtel Xstream Fiber", -1499, "essentials", CC),
        (18, "Jio Postpaid", -599, "essentials", CC),
        (7, "BESCOM Electricity", -random.randint(1750, 3400), "essentials", CC),
        (8, "BWSSB Water", -random.randint(380, 560), "essentials", SAV),
        (10, "Netflix India", -649, "subscription", CC),
        (9, "Spotify Premium", -149, "subscription", WAL),
        (4, "ChatGPT Plus", -1800, "subscription", CC),
        (12, "iCloud+ 200GB", -75, "subscription", CC),
        (15, "YouTube Premium", -149, "subscription", WAL),
        (5, "AWS Cloud Services", -random.randint(7800, 9600), "business", CC),
        (6, "Cult.fit Membership", -1500, "lifestyle", CC),
        (25, "Zerodha — ETF SIP", -10000, "investment", SAV),
        (2, "Vanguard S&P 500 SIP", -45000, "investment", SAV),
    ]
    for day, title, amt, cat, acct in fixed:
        d = clamp(date(y, m, day))
        if d:
            t = ts(d, 6, 21) if cat in ("subscription", "business") else ts(d)
            if acct == CC:
                cc_add(t, title, amt, cat)
            else:
                add(t, title, amt, cat, acct)

    # groceries 4–6×
    for _ in range(random.randint(4, 6)):
        d = clamp(date(y, m, random.randint(1, 28)))
        if d:
            cc_add(
                ts(d, 10, 21),
                f"{random.choice(GROCERY)} / Groceries",
                -random.randint(420, 4600),
                "essentials",
            )

    # food & coffee 9–14×
    for _ in range(random.randint(9, 14)):
        d = clamp(date(y, m, random.randint(1, 28)))
        if d:
            amt = -random.choice([183, 219, 268, 312, 345, 428, 516, 624, 745, 890, 1145, 1380])
            acct = WAL if random.random() < 0.5 else CC
            (cc_add if acct == CC else lambda t, ti, a, c: add(t, ti, a, c, WAL))(
                ts(d, 9, 23), random.choice(FOOD), amt, "lifestyle"
            )

    # transport 6–10×
    for _ in range(random.randint(6, 10)):
        d = clamp(date(y, m, random.randint(1, 28)))
        if d:
            add(
                ts(d, 8, 22),
                random.choice(RIDE),
                -random.choice([89, 124, 156, 189, 234, 312, 418, 540]),
                "lifestyle",
                WAL,
            )
    # fuel ~2×
    for _ in range(2):
        d = clamp(date(y, m, random.choice([6, 19])))
        if d:
            cc_add(
                ts(d, 8, 20),
                "Indian Oil — COCO Indiranagar",
                -random.randint(1700, 2300),
                "essentials",
            )

    # shopping 2–4×
    for _ in range(random.randint(2, 4)):
        d = clamp(date(y, m, random.randint(2, 27)))
        if d:
            cc_add(
                ts(d, 11, 22),
                random.choice(SHOP),
                -random.choice([549, 899, 1249, 1899, 2450, 3299, 4799, 7490, 11990]),
                "lifestyle",
            )

    # entertainment 1–2×, health 1–2×
    for _ in range(random.randint(1, 2)):
        d = clamp(date(y, m, random.randint(3, 27)))
        if d:
            cc_add(
                ts(d, 17, 23),
                random.choice(FUN),
                -random.choice([350, 520, 760, 999, 1450]),
                "lifestyle",
            )
    for _ in range(random.randint(1, 2)):
        d = clamp(date(y, m, random.randint(2, 27)))
        if d:
            add(
                ts(d, 9, 20),
                random.choice(HEALTH),
                -random.choice([260, 385, 540, 820, 1240]),
                "essentials",
                WAL,
            )

    # UPI splits with friends — both directions
    for _ in range(random.randint(2, 4)):
        d = clamp(date(y, m, random.randint(1, 28)))
        if d:
            f = random.choice(FRIENDS)
            if random.random() < 0.45:
                add(
                    ts(d, 10, 23),
                    f"UPI from {f} — split settle",
                    random.choice([420, 650, 845, 1250, 1600]),
                    "transfer",
                    WAL,
                )
            else:
                add(
                    ts(d, 10, 23),
                    f"UPI to {f} — dinner split",
                    -random.choice([380, 560, 720, 980, 1340]),
                    "transfer",
                    WAL,
                )

    # occasional crypto top-up
    if random.random() < 0.35:
        d = clamp(date(y, m, random.randint(8, 24)))
        if d:
            add(
                ts(d, 19, 23),
                random.choice(["WazirX — BTC buy", "CoinDCX — ETH buy", "CoinDCX — SOL buy"]),
                -random.choice([5000, 8000, 12000, 15000]),
                "investment",
                SAV,
            )

    # NOTE: no separate card-settlement row — the card spends above are the
    # expense records; adding the settlement debit too would double-count
    # every credit-card rupee in the frontend's net/heatmap math.

    cur = (date(y, m, 28) + timedelta(days=5)).replace(day=1)

# travel one-offs
for y, m, dd, title, amt, cat in TRAVEL:
    d = clamp(date(y, m, dd))
    if d:
        add(ts(d, 7, 21), title, amt, cat, CC)

# today (2026-06-11) — a normal morning so "Today" panels are alive
add(datetime(2026, 6, 11, 9, 12), "Blue Tokai Coffee", -240, "lifestyle", WAL)
add(datetime(2026, 6, 11, 9, 48), "Uber", -184, "lifestyle", WAL)
add(datetime(2026, 6, 11, 13, 22), "Swiggy", -326, "lifestyle", CC)

# ── fresh June trades at current snapshot prices ─────────────────────────────
snap = {
    r["symbol"]: float(r["price"])
    for r in M.query(
        "SELECT s.symbol, s.price FROM price_snapshots s "
        "JOIN (SELECT symbol, MAX(ts) ts FROM price_snapshots GROUP BY symbol) m "
        "ON s.symbol=m.symbol AND s.ts=m.ts"
    )
}
names = {
    r["symbol"]: r["name"]
    for r in M.query("SELECT symbol, name FROM asset_catalog WHERE category='stocks'")
}

JUNE_TRADES = [  # (day, hour, minute, side, symbol, qty)
    (2, 11, 42, "BUY", "NVDA", 6),
    (3, 14, 18, "BUY", "MSFT", 3),
    (4, 10, 55, "SELL", "AAPL", 4),
    (5, 15, 31, "BUY", "GOOGL", 4),
    (8, 12, 7, "SELL", "AMD", 3),
    (9, 13, 49, "BUY", "JPM", 5),
    (10, 11, 26, "BUY", "META", 1),
    (11, 10, 33, "BUY", "TSLA", 2),
]

trade_rows = []
for dd, hh, mm, side, sym, qty in JUNE_TRADES:
    base = snap.get(sym)
    if not base:
        continue
    px = round(base * random.uniform(0.985, 1.015), 2)  # near-market fill
    trade_rows.append(
        (
            USER,
            sym,
            names.get(sym, sym),
            side,
            qty,
            px,
            round(qty * px, 2),
            "USD",
            datetime(2026, 6, dd, hh, mm),
        )
    )

# ── write everything in one go ───────────────────────────────────────────────
with M.get_conn() as conn, conn.cursor() as cur_:
    cur_.execute("DELETE FROM transactions WHERE user_id=%s", (USER,))
    cur_.executemany(
        "INSERT INTO transactions (user_id, ext_id, title, tx_date, amount, category, account, tx_type) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
        [(USER, *r) for r in rows],
    )
    # only the June top-up window is regenerated — history before it stays
    cur_.execute("DELETE FROM trades WHERE user_id=%s AND trade_date >= '2026-06-02'", (USER,))
    cur_.executemany(
        "INSERT INTO trades (user_id, symbol, name, side, quantity, price, amount, currency, trade_date) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        trade_rows,
    )
    # balances retold to match the story (salary just landed, card mid-cycle)
    cur_.execute("UPDATE accounts SET balance=287450 WHERE user_id=%s AND name=%s", (USER, SAV))
    cur_.execute("UPDATE accounts SET balance=63120  WHERE user_id=%s AND name=%s", (USER, CC))
    cur_.execute("UPDATE accounts SET balance=16840  WHERE user_id=%s AND name=%s", (USER, WAL))

print(f"inserted {len(rows)} transactions ({START} → {TODAY}), {len(trade_rows)} June trades")
print("monthly density:")
for r in M.query(
    "SELECT DATE_FORMAT(tx_date,'%%Y-%%m') m, COUNT(*) n, ROUND(SUM(amount)) net FROM transactions WHERE user_id=1 GROUP BY m ORDER BY m"
):
    print("  ", r["m"], r["n"], "tx  net ₹", r["net"])
