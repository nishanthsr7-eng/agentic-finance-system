"""users.password_hash (was ensure_auth_schema)."""

from backend.migrate import add_column_if_missing


def up(cur):
    add_column_if_missing(cur, "users", "password_hash", "VARCHAR(255) NULL")
