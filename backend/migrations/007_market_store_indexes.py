"""Market store in TiDB (MARKET_STORE=mysql): make sure the indexes its queries
use exist, whatever DDL version first created the tables.

predictions / prediction_outcomes / calibration_buckets / ohlcv_history come
from 001_baseline; this adds the target_date index for get_due_predictions,
re-checks the ones the baseline should already have, and the iv_* columns that
were added to predictions later. MySQL has no CREATE INDEX IF NOT EXISTS, so
each one is guarded by SHOW INDEX.
"""

import logging

from backend.migrate import add_column_if_missing

log = logging.getLogger("flux.migrate")

INDEXES = (
    ("predictions", "idx_pred_sym", "(symbol, generated_at DESC)"),
    ("predictions", "idx_pred_target", "(target_date)"),
    ("trades", "idx_trade_user_date", "(user_id, trade_date DESC)"),
    ("market_alerts", "idx_alert_user", "(user_id, active)"),
)


def create_index_if_missing(cur, table: str, name: str, cols: str) -> bool:
    cur.execute(f"SHOW INDEX FROM `{table}` WHERE Key_name = %s", (name,))
    if cur.fetchall():
        return False
    cur.execute(f"CREATE INDEX `{name}` ON `{table}` {cols}")
    log.info("migration: index %s.%s created", table, name)
    return True


def up(cur):
    add_column_if_missing(cur, "predictions", "iv_atm", "DOUBLE")
    add_column_if_missing(cur, "predictions", "iv_skew", "DOUBLE")
    for table, name, cols in INDEXES:
        create_index_if_missing(cur, table, name, cols)
