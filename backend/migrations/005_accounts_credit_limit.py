"""accounts.credit_limit: available-limit figure for cards (was ensure_payments_schema)."""

from backend.migrate import add_column_if_missing


def up(cur):
    add_column_if_missing(cur, "accounts", "credit_limit", "DECIMAL(16,2) NOT NULL DEFAULT 0")
