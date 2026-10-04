"""trades.asset_type: tells crypto from stocks without a catalog join (was ensure_trading_schema)."""

from backend.migrate import add_column_if_missing


def up(cur):
    add_column_if_missing(cur, "trades", "asset_type", "VARCHAR(20) DEFAULT 'stocks'")
