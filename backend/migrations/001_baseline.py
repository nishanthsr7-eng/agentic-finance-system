"""Every table in mysql_db.DDL (CREATE TABLE IF NOT EXISTS), so an empty
database boots complete. A no-op on a database seeded by seed_mysql.py."""


def up(cur):
    from backend.mysql_db import DDL

    for stmt in DDL:
        cur.execute(stmt)
