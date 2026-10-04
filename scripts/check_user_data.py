"""Quick realism audit of user 1's seeded data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import mysql_db as M

print(
    "TX range:",
    M.query_one(
        "SELECT MIN(tx_date) a, MAX(tx_date) b, COUNT(*) n FROM transactions WHERE user_id=1"
    ),
)
print("\nTX last 10:")
for r in M.query(
    "SELECT tx_date, title, amount, category, account FROM transactions WHERE user_id=1 ORDER BY tx_date DESC LIMIT 10"
):
    print("  ", r)
print("\nTX by month:")
for r in M.query(
    "SELECT DATE_FORMAT(tx_date,'%%Y-%%m') m, COUNT(*) n, ROUND(SUM(amount)) s FROM transactions WHERE user_id=1 GROUP BY m ORDER BY m DESC LIMIT 8"
):
    print("  ", r)
print(
    "\nAccounts:", M.query("SELECT name, acct_type, balance, active FROM accounts WHERE user_id=1")
)
print(
    "\nTrades range:",
    M.query_one(
        "SELECT MIN(trade_date) a, MAX(trade_date) b, COUNT(*) n FROM trades WHERE user_id=1"
    ),
)
print("Trades last 5:")
for r in M.query(
    "SELECT trade_date, side, symbol, quantity, price, amount, currency FROM trades WHERE user_id=1 ORDER BY trade_date DESC LIMIT 5"
):
    print("  ", r)
print(
    "\nRecurring:",
    M.query("SELECT title, amount, due_day, category FROM recurring_payments WHERE user_id=1"),
)
print(
    "\nVault goals:",
    M.query("SELECT name, target, saved, monthly, deadline FROM vault_goals WHERE user_id=1"),
)
print("\nPortfolio:", M.query_one("SELECT * FROM portfolio WHERE user_id=1"))
print("\nContacts:", M.query("SELECT name, flux_id, favorite FROM contacts WHERE user_id=1"))
print("\nCatalog sample:", M.query("SELECT symbol, name, category FROM asset_catalog LIMIT 8"))
print(
    "\nTX categories:",
    M.query("SELECT category, COUNT(*) n FROM transactions WHERE user_id=1 GROUP BY category"),
)
