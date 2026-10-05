"""Rolling demo-user seed: dates follow `today`, balances add up, only user 1 is written.

No database: the row builder is pure, and the writer runs against a fake connection.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from backend import demo_seed as S

TODAY = date(2027, 1, 15)
NOW = datetime(2027, 1, 15, 18, 0)


def _closes(today: date = TODAY) -> dict[str, dict[str, float]]:
    out = {}
    for i, sym in enumerate(S.TRADE_SYMBOLS):
        out[sym] = {(today - timedelta(days=k)).isoformat(): 100.0 + 10 * i + k for k in range(40)}
    return out


@pytest.fixture(scope="module")
def data():
    return S.build_demo_rows(TODAY, NOW, _closes())


def test_transactions_end_today_and_never_in_the_future(data):
    dates = [t[2] for t in data["transactions"]]
    assert max(dates).startswith("2027-01-15")
    assert max(dates) <= "2027-01-15 18:00:00"
    assert min(dates) >= (TODAY - timedelta(days=365)).isoformat()


def test_goal_deadlines_are_in_the_future(data):
    assert all(g[5] > TODAY for g in data["goals"])


def test_dated_rows_follow_today(data):
    assert data["credit_as_of"] == date(2027, 1, 1)
    assert data["credit_history"][-1][0] == "2027-01"
    assert all(d <= TODAY and d.day == 1 for _n, _a, d, _note in data["deposits"])
    assert data["devices"][0][4] == "2027-01-15 18:00:00"  # current session = now
    assert data["events"][0][4] == "2027-01-15 18:00:00"  # last login = now
    assert all("2027-01-01" <= d[4] for d in data["devices"])  # all within the last 14 days


def test_no_nifty_anywhere(data):
    assert "NIFTY" not in repr(data).upper()


def test_balances_equal_the_transaction_sums(data):
    txs = data["transactions"]
    by_acct = lambda name: sum(t[3] for t in txs if t[5] == name)  # noqa: E731
    bal = data["balances"]
    assert bal[S.SAV] == pytest.approx(S.OPENING[S.SAV] + by_acct(S.SAV), abs=0.01)
    assert bal[S.WAL] == pytest.approx(S.OPENING[S.WAL] + by_acct(S.WAL), abs=0.01)
    cycle = sum(t[3] for t in txs if t[5] == S.CC and t[2] >= "2027-01-01")
    assert bal[S.CC] == pytest.approx(S.CC_LIMIT + cycle, abs=0.01)
    assert all(b > 0 for b in bal.values())
    assert {a[0]: a[2] for a in data["accounts"]} == bal


def test_same_date_gives_identical_rows():
    assert S.build_demo_rows(TODAY, NOW, _closes()) == S.build_demo_rows(TODAY, NOW, _closes())


def test_past_months_stay_put_from_one_day_to_the_next():
    a = S.build_transactions(TODAY, NOW)
    b = S.build_transactions(TODAY + timedelta(days=1), NOW + timedelta(days=1))
    dec = lambda rows: [r for r in rows if r[0].strftime("%Y-%m") == "2026-12"]  # noqa: E731
    assert dec(a) and dec(a) == dec(b)


def test_trades_use_stored_closes_and_never_oversell(data):
    trades = data["trades"]
    assert 1 <= len(trades) <= 15
    closes = _closes()
    held = dict.fromkeys(S.TRADE_SYMBOLS, 0)
    cash = S.SEED_CASH_USD
    for sym, _name, side, qty, price, amount, when in trades:
        assert sym in S.TRADE_SYMBOLS
        assert TODAY - timedelta(days=30) <= when.date() < TODAY
        assert price == round(closes[sym][when.date().isoformat()], 2)
        held[sym] += qty if side == "BUY" else -qty
        assert held[sym] >= 0
        cash += -amount if side == "BUY" else amount
    assert data["wallet_cash"] == pytest.approx(cash, abs=0.01)


def test_at_most_one_trade_per_symbol_per_day():
    for offset in range(60):  # many seeds, since the trade plan depends on the date
        today = TODAY - timedelta(days=offset)
        now = datetime(today.year, today.month, today.day, 21, 0)
        trades = S.build_demo_rows(today, now, _closes(today))["trades"]
        keys = [(sym, when.date()) for sym, _n, _side, _q, _p, _a, when in trades]
        assert len(keys) == len(set(keys))


def test_no_stored_closes_means_no_trades():
    data = S.build_demo_rows(TODAY, NOW, {})
    assert data["trades"] == []
    assert data["wallet_cash"] == S.SEED_CASH_USD


# ── writer ───────────────────────────────────────────────────────────────────


class _Cur:
    def __init__(self, log):
        self.log, self.lastrowid = log, 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        self.lastrowid += 1
        self.log.append((sql, [tuple(params)]))

    def executemany(self, sql, rows):
        self.log.append((sql, [tuple(r) for r in rows]))


class _Conn:
    def __init__(self, fail_on: str | None = None):
        self.log, self.events, self.fail_on = [], [], fail_on

    def cursor(self):
        cur = _Cur(self.log)
        if self.fail_on:
            real = cur.execute

            def execute(sql, params=()):
                if self.fail_on in sql:
                    raise RuntimeError("boom")
                real(sql, params)

            cur.execute = execute
        return cur

    def begin(self):
        self.events.append("begin")

    def commit(self):
        self.events.append("commit")

    def rollback(self):
        self.events.append("rollback")


def test_writer_only_touches_user_1_in_one_transaction(data):
    conn = _Conn()
    S.write_demo_rows(conn, data, TODAY)
    assert conn.events == ["begin", "commit"]
    deleted = {sql.split("`")[1] for sql, _ in conn.log if sql.startswith("DELETE")}
    assert deleted == set(S.USER_TABLES)
    for sql, rows in conn.log:
        if sql.startswith("DELETE"):
            assert "WHERE user_id=%s" in sql and rows == [(1,)]
        elif "app_state" not in sql:
            assert rows and all(r[0] == 1 for r in rows), sql


def test_writer_rolls_back_on_error(data):
    conn = _Conn(fail_on="INSERT INTO portfolio ")
    with pytest.raises(RuntimeError):
        S.write_demo_rows(conn, data, TODAY)
    assert conn.events == ["begin", "rollback"]


def test_scheduler_registers_the_reseed_jobs():
    from backend import ingestion

    sched = ingestion.build_scheduler()
    ids = {j.id for j in sched.get_jobs()}
    assert {"demo_reseed", "demo_reseed_boot"} <= ids


def test_reseed_at_0535_ist_still_has_a_row_today():
    early = datetime(2027, 1, 15, 5, 35)
    rows = S.build_demo_rows(TODAY, early, _closes())["transactions"]
    assert max(t[2] for t in rows).startswith("2027-01-15")
    assert all(t[2] <= "2027-01-15 05:35:00" for t in rows)


def test_every_recent_day_has_spending():
    # The Analysis cashflow heatmap covers 30 days; none should be empty.
    for today in (TODAY, date(2026, 10, 4), date(2026, 3, 1)):
        now = datetime(today.year, today.month, today.day, 18, 0)
        rows = S.build_transactions(today, now)
        spent = {r[0].date() for r in rows if r[2] < 0}
        missing = [
            today - timedelta(days=k) for k in range(30) if today - timedelta(days=k) not in spent
        ]
        assert missing == []


def test_allocation_matches_holdings_and_cash(data):
    equity, crypto, cash, _goal = data["portfolio"]
    assert equity + crypto + cash == 100
    cost = {"equity": 0.0, "crypto": 0.0}
    for _s, _n, kind, qty, avg, _c in data["holdings"]:
        cost["crypto" if kind == "crypto" else "equity"] += qty * avg
    cash_total = sum(a[2] for a in data["accounts"] if a[1] in ("savings", "wallet"))
    assert crypto == round(cost["crypto"] / (sum(cost.values()) + cash_total) * 100)
