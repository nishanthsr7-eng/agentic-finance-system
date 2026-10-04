import os
import sqlite3

db_path = "ai_engine/cache/flux.db"

if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
c = conn.cursor()

# Check tables
c.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = c.fetchall()
print(f"Tables: {[t[0] for t in tables]}")

for table in [t[0] for t in tables]:
    c.execute(f"SELECT COUNT(*) FROM {table}")
    count = c.fetchone()[0]
    print(f"Table {table}: {count} rows")

    if table == "trade_journal" and count > 0:
        c.execute("SELECT * FROM trade_journal LIMIT 5")
        rows = c.fetchall()
        print(f"Sample trades: {rows}")

conn.close()
