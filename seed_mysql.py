"""
FLUX — Ultimate MySQL Seed
==========================

Creates the `flux` MySQL database, builds every table (via backend.mysql_db),
and fills it with the data each page of the app needs:

  Personal finance (demo user "Nishanth", id=1)
    • accounts, ~12 months of transactions, portfolio + holdings,
      recurring payments, contacts, vault goals + deposits,
      credit score + factors + 12-month history, rewards,
      security settings, devices, security events
  Site content
    • FAQs, job openings, team members
  Market data
    • asset_catalog + a full port of every market table from the existing
      SQLite store (flux_market.db) so /db market endpoints match the live ones.

Run
---
    python seed_mysql.py                 # full seed (ports OHLCV history too)
    python seed_mysql.py --skip-history  # skip the large ohlcv_history table
    python seed_mysql.py --no-market     # personal-finance + content only

Requires MYSQL_* values in .env (MYSQL_PASSWORD especially).
"""

from __future__ import annotations

import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

# Make `backend` importable when run from the project root.
sys.path.insert(0, str(Path(__file__).parent))

from backend import mysql_db as M  # noqa: E402

ROOT = Path(__file__).parent
SQLITE_PATH = ROOT / "flux_market.db"
USER_ID = 1
NOW_MS = int(time.time() * 1000)


# ── helpers ──────────────────────────────────────────────────────────────────


def executemany(cur, sql: str, rows: list[tuple]) -> int:
    if not rows:
        return 0
    cur.executemany(sql, rows)
    return len(rows)


def wipe(cur) -> None:
    """Clear all rows so re-seeding is idempotent (children first)."""
    tables = [
        # personal finance children → parents
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
        # content
        "faqs",
        "job_openings",
        "team_members",
        # market
        "price_snapshots",
        "ohlcv_daily",
        "ohlcv_history",
        "ai_insights",
        "news_cache",
        "predictions",
        "prediction_outcomes",
        "calibration_buckets",
        "news_sentiment",
        "options_iv",
        "ingestion_log",
        "asset_catalog",
        # finally users
        "users",
    ]
    cur.execute("SET FOREIGN_KEY_CHECKS=0")
    for t in tables:
        cur.execute(f"TRUNCATE TABLE `{t}`")
    cur.execute("SET FOREIGN_KEY_CHECKS=1")


# ── 1. Personal finance ────────────────────────────────────────────────────────


def seed_user(cur) -> None:
    cur.execute(
        "INSERT INTO users (id, name, email, flux_id, phone, avatar, plan, currency, city) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (
            USER_ID,
            "Nishanth",
            "nishanth@flux.app",
            "nishanth@flux",
            "+91 98860 12345",
            "https://i.pravatar.cc/150?img=12",
            "FLUX Black",
            "INR",
            "Bengaluru",
        ),
    )


def seed_accounts(cur) -> None:
    rows = [
        (
            USER_ID,
            "HDFC Platinum",
            "credit",
            84500.00,
            "•••• •••• •••• 8842",
            "4532 9982 1104 8842",
            "••/••",
            "08/29",
            1,
            0,
        ),
        (
            USER_ID,
            "ICICI Wealth",
            "savings",
            245000.00,
            "•••• •••• •••• 1290",
            "5104 2293 8810 1290",
            "••/••",
            "03/28",
            0,
            1,
        ),
        (
            USER_ID,
            "FLUX Wallet",
            "wallet",
            18650.00,
            "•••• •••• •••• 4471",
            "6011 5566 7788 4471",
            "••/••",
            "12/27",
            0,
            2,
        ),
    ]
    executemany(
        cur,
        "INSERT INTO accounts (user_id,name,acct_type,balance,card_masked,card_real,"
        "expiry_masked,expiry_real,active,sort_order) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        rows,
    )


def seed_transactions(cur) -> None:
    """~12 months of realistic transactions anchored to today (mirrors js/seed.js, richer)."""
    salary_base = [182000, 184500, 188000]
    grocery_amt = [4800, 5600, 6200]
    rest_amt = [2200, 3100, 4000]
    zomato_amt = [1200, 1900, 2600]
    uber_amt = [600, 950, 1400]
    invest_extra = [0, 0, 25000]

    rows: list[tuple] = []
    counter = 1

    def add(y, m, d, max_day, title, amount, category, account, tx_type):
        nonlocal counter
        if d > max_day:
            return
        ext = f"tx_{counter:04d}"
        counter += 1
        dt = datetime(y, m, d, 10, 0, 0)
        rows.append(
            (
                USER_ID,
                ext,
                title,
                dt.strftime("%Y-%m-%d %H:%M:%S"),
                amount,
                category,
                account,
                tx_type,
            )
        )

    def gen_month(y, m, v, max_day):
        add(
            y,
            m,
            1,
            max_day,
            "Salary — Acme Corp",
            salary_base[v],
            "income",
            "ICICI Wealth",
            "income",
        )
        add(
            y,
            m,
            2,
            max_day,
            "Vanguard S&P 500 SIP",
            -45000,
            "investment",
            "HDFC Platinum",
            "expense",
        )
        add(
            y,
            m,
            3,
            max_day,
            "BigBasket / Groceries",
            -grocery_amt[v],
            "essentials",
            "HDFC Platinum",
            "expense",
        )
        add(y, m, 4, max_day, "ChatGPT Plus", -1800, "subscription", "HDFC Platinum", "expense")
        add(y, m, 5, max_day, "AWS Cloud Services", -8500, "business", "HDFC Platinum", "expense")
        add(y, m, 7, max_day, "BESCOM Electricity", -2800, "essentials", "HDFC Platinum", "expense")
        add(y, m, 9, max_day, "Spotify Premium", -149, "subscription", "FLUX Wallet", "expense")
        add(y, m, 10, max_day, "Netflix India", -649, "subscription", "HDFC Platinum", "expense")
        add(
            y,
            m,
            12,
            max_day,
            "Restaurant — Taj Bistro",
            -rest_amt[v],
            "lifestyle",
            "HDFC Platinum",
            "expense",
        )
        add(
            y,
            m,
            14,
            max_day,
            "Adobe Creative Cloud",
            -1450,
            "subscription",
            "HDFC Platinum",
            "expense",
        )
        add(
            y,
            m,
            15,
            max_day,
            "Zomato / Blinkit",
            -zomato_amt[v],
            "lifestyle",
            "FLUX Wallet",
            "expense",
        )
        if invest_extra[v]:
            add(
                y,
                m,
                18,
                max_day,
                "Binance — BTC Purchase",
                -invest_extra[v],
                "investment",
                "ICICI Wealth",
                "expense",
            )
        add(y, m, 20, max_day, "Uber / Rapido", -uber_amt[v], "lifestyle", "FLUX Wallet", "expense")
        if v == 2:
            add(
                y, m, 20, max_day, "Freelance Settlement", 18000, "income", "ICICI Wealth", "income"
            )
        add(
            y, m, 22, max_day, "Rent — Koramangala", -22000, "essentials", "ICICI Wealth", "expense"
        )
        add(y, m, 25, max_day, "Zerodha — ETF SIP", -10000, "investment", "ICICI Wealth", "expense")
        add(y, m, 28, max_day, "HDFC CC Settlement", -12000, "transfer", "ICICI Wealth", "transfer")

    today = date.today()
    cur_y, cur_m, cur_d = today.year, today.month, today.day

    def last_day(y, m):
        nm = date(y + (m // 12), (m % 12) + 1, 1)
        return (nm - timedelta(days=1)).day

    # 11 complete prior months
    for i in range(11, 0, -1):
        # step back i months
        y, m = cur_y, cur_m
        total = (cur_y * 12 + (cur_m - 1)) - i
        y, m = divmod(total, 12)
        m += 1
        v = (11 - i) % 3
        gen_month(y, m, v, last_day(y, m))

    # current partial month
    gen_month(cur_y, cur_m, 11 % 3, cur_d)

    executemany(
        cur,
        "INSERT INTO transactions (user_id,ext_id,title,tx_date,amount,category,account,tx_type) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
        rows,
    )
    return len(rows)


def seed_portfolio(cur) -> None:
    cur.execute(
        "INSERT INTO portfolio (user_id,equity_pct,crypto_pct,cash_pct,goal) VALUES (%s,%s,%s,%s,%s)",
        (USER_ID, 45, 30, 25, 15000000),
    )
    holdings = [
        (USER_ID, "SPY", "SPDR S&P 500 ETF", "etf", 2, 48900.00, "USD"),
        (USER_ID, "RELIANCE", "Reliance Industries", "equity", 35, 2890.00, "INR"),
        (USER_ID, "TCS", "Tata Consultancy", "equity", 18, 3960.00, "INR"),
        (USER_ID, "INFY", "Infosys Ltd", "equity", 60, 1540.00, "INR"),
        (USER_ID, "BTC", "Bitcoin", "crypto", 0.085, 5400000, "INR"),
        (USER_ID, "ETH", "Ethereum", "crypto", 1.4, 280000, "INR"),
        (USER_ID, "SOL", "Solana", "crypto", 22, 14500, "INR"),
    ]
    executemany(
        cur,
        "INSERT INTO portfolio_holdings (user_id,symbol,name,asset_type,quantity,avg_price,currency) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s)",
        holdings,
    )


def seed_recurring(cur) -> None:
    rows = [
        (USER_ID, "AWS Infrastructure", 8500, 5, "business", 1),
        (USER_ID, "ChatGPT Plus", 1800, 4, "subscription", 1),
        (USER_ID, "Netflix India", 649, 10, "subscription", 1),
        (USER_ID, "Spotify Premium", 149, 9, "subscription", 1),
        (USER_ID, "Rent — Koramangala", 22000, 22, "essentials", 1),
        (USER_ID, "Zerodha — ETF SIP", 10000, 25, "investment", 1),
    ]
    executemany(
        cur,
        "INSERT INTO recurring_payments (user_id,title,amount,due_day,category,active) "
        "VALUES (%s,%s,%s,%s,%s,%s)",
        rows,
    )


def seed_contacts(cur) -> None:
    rows = [
        (USER_ID, "Alex", "alex@flux", "A", 1),
        (USER_ID, "Sanjay", "sanjay@flux", "S", 1),
        (USER_ID, "Meera", "meera@flux", "M", 0),
        (USER_ID, "Priya", "priya@flux", "P", 0),
        (USER_ID, "Rohan", "rohan@flux", "R", 0),
    ]
    executemany(
        cur,
        "INSERT INTO contacts (user_id,name,flux_id,initial,favorite) VALUES (%s,%s,%s,%s,%s)",
        rows,
    )


def seed_vault(cur) -> None:
    goals = [
        (USER_ID, "Emergency Fund", "🛡️", 300000, 87500, 8000, "2026-12-31"),
        (USER_ID, "Europe Trip", "✈️", 150000, 42000, 5000, "2026-11-01"),
        (USER_ID, "New MacBook", "📱", 180000, 162000, 15000, "2026-07-01"),
        (USER_ID, "Down Payment", "🏠", 2500000, 320000, 25000, "2030-01-01"),
    ]
    cur.executemany(
        "INSERT INTO vault_goals (user_id,name,icon,target,saved,monthly,deadline) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s)",
        goals,
    )
    # map goal name → id for deposits
    cur.execute("SELECT id, name FROM vault_goals WHERE user_id=%s", (USER_ID,))
    gid = {r["name"]: r["id"] for r in cur.fetchall()}
    deposits = [
        (
            USER_ID,
            gid["Emergency Fund"],
            "Emergency Fund",
            8000,
            "2026-06-01",
            "Monthly auto-deposit",
        ),
        (USER_ID, gid["Europe Trip"], "Europe Trip", 5000, "2026-05-28", "Saved from bonus"),
        (USER_ID, gid["New MacBook"], "New MacBook", 15000, "2026-05-15", "Monthly contribution"),
        (
            USER_ID,
            gid["Emergency Fund"],
            "Emergency Fund",
            8000,
            "2026-05-01",
            "Monthly auto-deposit",
        ),
        (USER_ID, gid["Down Payment"], "Down Payment", 25000, "2026-04-30", "Monthly contribution"),
        (USER_ID, gid["Europe Trip"], "Europe Trip", 5000, "2026-04-28", "Monthly contribution"),
    ]
    cur.executemany(
        "INSERT INTO vault_deposits (user_id,goal_id,goal_name,amount,dep_date,note) "
        "VALUES (%s,%s,%s,%s,%s,%s)",
        deposits,
    )


def seed_credit(cur) -> None:
    cur.execute(
        "INSERT INTO credit_scores (user_id,score,max_score,rating,bureau,updated_at) "
        "VALUES (%s,%s,%s,%s,%s,%s)",
        (USER_ID, 762, 900, "Excellent", "CIBIL", "2026-06-01"),
    )
    factors = [
        (USER_ID, "Payment History", "Excellent", "100% on-time", 35, "High", 0),
        (USER_ID, "Credit Utilization", "Good", "24%", 30, "High", 1),
        (USER_ID, "Credit Age", "Good", "6 yr 4 mo", 15, "Medium", 2),
        (USER_ID, "Account Mix", "Excellent", "4 types", 10, "Low", 3),
        (USER_ID, "Hard Inquiries", "Fair", "3 in 12 mo", 10, "Medium", 4),
    ]
    cur.executemany(
        "INSERT INTO credit_factors (user_id,factor,status,value_txt,weight_pct,impact,sort_order) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s)",
        factors,
    )
    # 12-month score trend ending at 762
    today = date.today()
    scores = [712, 718, 724, 729, 733, 738, 742, 747, 751, 755, 759, 762]
    hist = []
    for i, sc in enumerate(scores):
        total = (today.year * 12 + (today.month - 1)) - (11 - i)
        y, m = divmod(total, 12)
        hist.append((USER_ID, f"{y}-{m + 1:02d}", sc))
    cur.executemany(
        "INSERT INTO credit_history (user_id,month,score) VALUES (%s,%s,%s)",
        hist,
    )


def seed_rewards(cur) -> None:
    rows = [
        (USER_ID, "first_payment", "First Payment Sent", 250, 1),
        (USER_ID, "vault_starter", "Opened a Vault Goal", 150, 1),
        (USER_ID, "streak_30", "30-Day Activity Streak", 500, 1),
        (USER_ID, "invest_5l", "₹5L Invested Milestone", 1000, 0),
        (USER_ID, "referral_3", "Referred 3 Friends", 750, 0),
        (USER_ID, "credit_750", "Crossed 750 Credit Score", 600, 1),
    ]
    cur.executemany(
        "INSERT INTO rewards (user_id,reward_key,title,points,claimed) VALUES (%s,%s,%s,%s,%s)",
        rows,
    )


def seed_security(cur) -> None:
    settings_rows = [
        (USER_ID, "2fa", "Two-Factor Authentication", 1, 0),
        (USER_ID, "biometric", "Biometric Unlock", 1, 1),
        (USER_ID, "txn_alerts", "Transaction Alerts", 1, 2),
        (USER_ID, "login_alerts", "New Login Alerts", 1, 3),
        (USER_ID, "intl_block", "Block International Cards", 0, 4),
        (USER_ID, "spend_limit", "Daily Spend Limit", 0, 5),
    ]
    cur.executemany(
        "INSERT INTO security_settings (user_id,setting_key,label,enabled,sort_order) "
        "VALUES (%s,%s,%s,%s,%s)",
        settings_rows,
    )
    devices = [
        (
            USER_ID,
            "iPhone 15 Pro",
            "iOS 18",
            "Safari",
            "Bengaluru, IN",
            "2026-06-10 09:14:00",
            1,
            1,
        ),
        (
            USER_ID,
            'MacBook Pro 14"',
            "macOS 15",
            "Chrome",
            "Bengaluru, IN",
            "2026-06-10 08:02:00",
            1,
            0,
        ),
        (
            USER_ID,
            "Windows Desktop",
            "Windows 11",
            "Edge",
            "Bengaluru, IN",
            "2026-06-08 21:40:00",
            1,
            0,
        ),
        (USER_ID, "iPad Air", "iPadOS 18", "Safari", "Mumbai, IN", "2026-05-29 18:20:00", 0, 0),
    ]
    cur.executemany(
        "INSERT INTO devices (user_id,device_name,os,browser,location,last_active,trusted,current_session) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
        devices,
    )
    events = [
        (USER_ID, "login", "Successful login", "Bengaluru, IN", "info", "2026-06-10 09:14:00"),
        (
            USER_ID,
            "alert",
            "Large transaction flagged ₹45,000",
            "Bengaluru, IN",
            "warning",
            "2026-06-02 10:01:00",
        ),
        (
            USER_ID,
            "password_change",
            "Password changed",
            "Bengaluru, IN",
            "info",
            "2026-05-20 14:30:00",
        ),
        (
            USER_ID,
            "login",
            "Blocked login attempt",
            "Unknown, RU",
            "critical",
            "2026-05-12 03:11:00",
        ),
        (
            USER_ID,
            "device",
            "New device added: iPad Air",
            "Mumbai, IN",
            "info",
            "2026-05-29 18:20:00",
        ),
    ]
    cur.executemany(
        "INSERT INTO security_events (user_id,event_type,description,location,severity,event_at) "
        "VALUES (%s,%s,%s,%s,%s,%s)",
        events,
    )


# Marketplace equities (must match backend STOCK_META) with rough USD prices for
# realistic trade fills. Only these symbols ever appear in the Live Ledger.
MARKETPLACE_STOCKS = [
    ("AAPL", "Apple Inc.", 210.0),
    ("MSFT", "Microsoft Corp.", 440.0),
    ("NVDA", "NVIDIA Corp.", 135.0),
    ("GOOGL", "Alphabet Inc.", 180.0),
    ("AMZN", "Amazon.com Inc.", 200.0),
    ("TSLA", "Tesla Inc.", 250.0),
    ("META", "Meta Platforms", 580.0),
    ("NFLX", "Netflix Inc.", 700.0),
    ("JPM", "JPMorgan Chase", 210.0),
    ("AMD", "Advanced Micro Dev.", 160.0),
    ("TSM", "Taiwan Semiconductor", 190.0),
    ("ORCL", "Oracle Corp.", 170.0),
    ("CRM", "Salesforce Inc.", 280.0),
    ("INTC", "Intel Corp.", 30.0),
    ("BABA", "Alibaba Group", 95.0),
]


def seed_trades(cur) -> int:
    """Realistic BUY/SELL trade history for marketplace stocks over the past ~12 months.

    Deterministic (fixed RNG seed) so re-seeding is stable. Each symbol opens with a
    BUY and accumulates a mix of follow-on buys and partial sells; quantity sold never
    exceeds the running position so the ledger reads like a real trade book.
    """
    import random

    rng = random.Random(42)

    today = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
    rows: list[tuple] = []

    for sym, name, base in MARKETPLACE_STOCKS:
        position = 0.0
        # 3–6 trades per symbol, spread across the last 360 days (oldest first)
        n = rng.randint(3, 6)
        day_offsets = sorted(rng.sample(range(5, 360), n), reverse=True)
        for i, off in enumerate(day_offsets):
            tdate = today - timedelta(days=off, hours=rng.randint(0, 5), minutes=rng.randint(0, 59))
            # price wiggles ±18% around the base
            price = round(base * rng.uniform(0.82, 1.18), 2)
            # First trade is always a BUY; later trades sell only if we hold shares
            if i == 0 or position <= 0 or rng.random() < 0.55:
                side = "BUY"
                qty = rng.randint(2, 25)
                position += qty
            else:
                side = "SELL"
                qty = rng.randint(1, max(1, int(position)))
                position -= qty
            amount = round(qty * price, 2)
            rows.append(
                (
                    USER_ID,
                    sym,
                    name,
                    side,
                    qty,
                    price,
                    amount,
                    "USD",
                    tdate.strftime("%Y-%m-%d %H:%M:%S"),
                )
            )

    # newest first not required (API orders), insert as built
    cur.executemany(
        "INSERT INTO trades (user_id,symbol,name,side,quantity,price,amount,currency,trade_date) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        rows,
    )
    return len(rows)


# ── 2. Site content ───────────────────────────────────────────────────────────


def seed_content(cur) -> None:
    faqs = [
        (
            "Account",
            "How do I open a FLUX account?",
            "Download the app, verify your phone and PAN, complete the 2-minute KYC, and your FLUX account is live instantly with a virtual card.",
        ),
        (
            "Account",
            "Is FLUX free to use?",
            "The core FLUX plan is free forever. FLUX Black (₹499/mo) adds higher limits, priority support, and metal card.",
        ),
        (
            "Security",
            "How is my money protected?",
            "Funds are held with RBI-regulated partner banks. We use 256-bit encryption, biometric unlock, and real-time fraud monitoring.",
        ),
        (
            "Security",
            "What happens if I lose my phone?",
            "Freeze your card instantly from any device or by calling support. Your account is protected by 2FA and biometric locks.",
        ),
        (
            "Payments",
            "Are there fees on transfers?",
            "UPI and FLUX-to-FLUX transfers are always free. Bank transfers above the free monthly quota carry a small flat fee.",
        ),
        (
            "Payments",
            "How fast are payments settled?",
            "FLUX-to-FLUX is instant. UPI and IMPS settle within seconds; NEFT follows standard banking windows.",
        ),
        (
            "Investing",
            "Can I invest through FLUX?",
            "Yes — buy ETFs, stocks, and crypto from the Advisor and Marketplace pages. SIPs can be automated from the dashboard.",
        ),
        (
            "Investing",
            "Is the AI advisor regulated financial advice?",
            "The AI advisor provides educational insights and is not SEBI-registered investment advice. Always do your own research.",
        ),
        (
            "Credit",
            "How is my credit score calculated?",
            "We surface your CIBIL score, refreshed monthly, broken down by payment history, utilization, age, mix, and inquiries.",
        ),
        (
            "Credit",
            "Does checking my score hurt it?",
            "No. Viewing your score in FLUX is a soft inquiry and never affects your credit rating.",
        ),
    ]
    cur.executemany(
        "INSERT INTO faqs (category,question,answer,sort_order) VALUES (%s,%s,%s,%s)",
        [(c, q, a, i) for i, (c, q, a) in enumerate(faqs)],
    )

    jobs = [
        (
            "Senior Backend Engineer",
            "Engineering",
            "Bengaluru, IN",
            "Full-time",
            "Own the core ledger and payments services. Python/Go, distributed systems, financial-grade reliability.",
        ),
        (
            "ML Engineer — Forecasting",
            "AI",
            "Remote, IN",
            "Full-time",
            "Ship the prediction engine: feature pipelines, model serving, calibration, and backtesting at scale.",
        ),
        (
            "Product Designer",
            "Design",
            "Bengaluru, IN",
            "Full-time",
            "Design the FLUX experience end-to-end — from dashboards to delight. Strong systems thinking and motion.",
        ),
        (
            "Data Scientist",
            "AI",
            "Remote, IN",
            "Full-time",
            "Turn market and behavioral data into signals. Time-series, NLP sentiment, and rigorous experimentation.",
        ),
        (
            "Frontend Engineer",
            "Engineering",
            "Bengaluru, IN",
            "Full-time",
            "Build the fast, glassy FLUX web app. Vanilla JS mastery, performance, and pixel-perfect execution.",
        ),
        (
            "Compliance Lead",
            "Operations",
            "Mumbai, IN",
            "Full-time",
            "Own regulatory relationships and KYC/AML programs across our banking partners.",
        ),
        (
            "Engineering Intern",
            "Engineering",
            "Bengaluru, IN",
            "Intern",
            "6-month internship across backend and data. Ship real features used by thousands.",
        ),
    ]
    cur.executemany(
        "INSERT INTO job_openings (title,department,location,emp_type,description,active) "
        "VALUES (%s,%s,%s,%s,%s,1)",
        jobs,
    )

    team = [
        (
            "Aarav Mehta",
            "Co-Founder & CEO",
            "Ex-fintech operator obsessed with money that works for you.",
            "https://i.pravatar.cc/150?img=11",
            0,
        ),
        (
            "Diya Sharma",
            "Co-Founder & CTO",
            "Built payments infra at scale. Leads engineering and AI.",
            "https://i.pravatar.cc/150?img=45",
            1,
        ),
        (
            "Kabir Nair",
            "Head of Design",
            "Crafts the FLUX feel — calm, fast, and trustworthy.",
            "https://i.pravatar.cc/150?img=33",
            2,
        ),
        (
            "Ananya Rao",
            "Head of Data Science",
            "Turns market noise into clear, calibrated signals.",
            "https://i.pravatar.cc/150?img=20",
            3,
        ),
    ]
    cur.executemany(
        "INSERT INTO team_members (name,role,bio,avatar,sort_order) VALUES (%s,%s,%s,%s,%s)",
        team,
    )


# ── 3. Market data (port from SQLite) ──────────────────────────────────────────

# (sqlite table, mysql table, column list) — columns match between stores.
MARKET_TABLES = [
    (
        "price_snapshots",
        "price_snapshots",
        ["symbol", "asset_type", "name", "price", "change_pct", "volume", "market_cap", "ts"],
    ),
    ("ohlcv_daily", "ohlcv_daily", ["symbol", "date", "open", "high", "low", "close", "volume"]),
    (
        "ai_insights",
        "ai_insights",
        ["symbol", "insight_type", "content", "sentiment", "confidence", "generated_at"],
    ),
    (
        "news_cache",
        "news_cache",
        ["title", "source", "url", "summary", "published_at", "cached_at"],
    ),
    (
        "predictions",
        "predictions",
        [
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
            "iv_atm",
            "iv_skew",
            "generated_at",
            "target_date",
        ],
    ),
    (
        "prediction_outcomes",
        "prediction_outcomes",
        ["prediction_id", "actual_return", "correct", "pnl_after_costs", "resolved_at"],
    ),
    (
        "calibration_buckets",
        "calibration_buckets",
        ["model", "bucket", "stated_conf", "realized_hit", "n", "updated_at"],
    ),
    (
        "news_sentiment",
        "news_sentiment",
        ["url", "symbol", "source", "title", "score", "label", "published_at", "scored_at"],
    ),
    (
        "options_iv",
        "options_iv",
        ["symbol", "date", "spot", "atm_iv", "skew", "term_slope", "n_contracts", "snapshot_at"],
    ),
    ("ingestion_log", "ingestion_log", ["job", "status", "rows", "message", "ts"]),
]


def port_market(cur, skip_history: bool) -> dict:
    if not SQLITE_PATH.exists():
        print(f"  ! {SQLITE_PATH.name} not found — skipping market port")
        return {}
    counts: dict[str, int] = {}
    sconn = sqlite3.connect(str(SQLITE_PATH))
    sconn.row_factory = sqlite3.Row
    try:
        tables = list(MARKET_TABLES)
        if not skip_history:
            tables.append(
                (
                    "ohlcv_history",
                    "ohlcv_history",
                    ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"],
                )
            )
        for s_tab, m_tab, cols in tables:
            try:
                src = sconn.execute(f"SELECT {','.join(cols)} FROM {s_tab}").fetchall()
            except sqlite3.OperationalError:
                continue
            placeholders = ",".join(["%s"] * len(cols))
            # backtick-quote `rows` (reserved-ish) and any column safely
            col_sql = ",".join(f"`{c}`" for c in cols)
            sql = f"INSERT INTO {m_tab} ({col_sql}) VALUES ({placeholders})"
            data = [tuple(r[c] for c in cols) for r in src]
            # batch insert
            B = 5000
            for i in range(0, len(data), B):
                cur.executemany(sql, data[i : i + B])
            counts[m_tab] = len(data)
            print(f"  · {m_tab:<22} {len(data):>7} rows")
    finally:
        sconn.close()
    return counts


def seed_asset_catalog(cur) -> None:
    """Distinct symbols from snapshots + curated forex/indices/commodities for marketplace."""
    cur.execute(
        "SELECT symbol, asset_type, name, MAX(ts) AS m FROM price_snapshots "
        "GROUP BY symbol, asset_type, name"
    )
    seen = cur.fetchall()
    rows = []
    for r in seen:
        sym = r["symbol"]
        atype = r["asset_type"] or ""
        category = (
            "crypto" if atype == "crypto" else ("stocks" if atype == "stock" else atype or "stocks")
        )
        sub = sym.split(":")[-1].replace("USDT", "") if ":" in sym else sym
        rows.append((sym, r["name"], sub, category, "", ""))
    # curated extras not in live snapshots
    extras = [
        ("EURUSD", "Euro / US Dollar", "EUR", "forex", "", ""),
        ("USDINR", "US Dollar / Rupee", "INR", "forex", "", ""),
        ("GBPUSD", "Pound / US Dollar", "GBP", "forex", "", ""),
        ("^BSESN", "BSE SENSEX", "SENSEX", "indices", "", ""),
        ("^GSPC", "S&P 500", "SPX", "indices", "", ""),
        ("GC=F", "Gold Futures", "GOLD", "commodities", "", ""),
        ("CL=F", "Crude Oil WTI", "OIL", "commodities", "", ""),
        ("SI=F", "Silver Futures", "SILVER", "commodities", "", ""),
    ]
    have = {r[0] for r in rows}
    rows += [e for e in extras if e[0] not in have]
    cur.executemany(
        "INSERT INTO asset_catalog (symbol,name,sub,category,sector,icon) "
        "VALUES (%s,%s,%s,%s,%s,%s) "
        "ON DUPLICATE KEY UPDATE name=VALUES(name), category=VALUES(category)",
        rows,
    )
    print(f"  · asset_catalog         {len(rows):>7} rows")


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    skip_history = "--skip-history" in sys.argv
    no_market = "--no-market" in sys.argv

    print("FLUX — Ultimate MySQL Seed")
    print("=" * 50)

    print("1) Initialising schema …")
    M.init_schema()

    with M.get_conn() as conn:
        conn.autocommit(False)
        with conn.cursor() as cur:
            print("2) Wiping existing rows …")
            wipe(cur)

            print("3) Seeding personal finance …")
            seed_user(cur)
            seed_accounts(cur)
            n_tx = seed_transactions(cur)
            seed_portfolio(cur)
            seed_recurring(cur)
            seed_contacts(cur)
            seed_vault(cur)
            seed_credit(cur)
            seed_rewards(cur)
            seed_security(cur)
            n_trades = seed_trades(cur)
            print(f"   · transactions          {n_tx:>7} rows")
            print(f"   · trades                 {n_trades:>7} rows")

            print("4) Seeding site content …")
            seed_content(cur)

            if not no_market:
                print("5) Porting market data from SQLite …")
                port_market(cur, skip_history)
                seed_asset_catalog(cur)
            else:
                print("5) Skipping market data (--no-market)")

        conn.commit()

    print("=" * 50)
    print(f"Done. Database `{M.settings.MYSQL_DB}` seeded.")


if __name__ == "__main__":
    main()
