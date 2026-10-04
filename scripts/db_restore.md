# Database backup and restore

The only persistent database is MySQL on TiDB Serverless. SQLite on Render is a
cache that refills itself within an hour, and Chroma rebuilds itself.

## Weekly backup (from your own machine)

```bash
python scripts/db_backup.py
```

This writes `~/flux-backups/flux-YYYYMMDD-HHMMSS.json.gz` using the MySQL settings
in `.env`. Use the admin user. The file contains password hashes, so never commit
it or put it in a shared folder. Keep the last few and delete the older ones.

## Restore into an existing database

```bash
python scripts/db_backup.py --restore ~/flux-backups/FILE.json.gz            # dry run
python scripts/db_backup.py --restore ~/flux-backups/FILE.json.gz --apply    # all tables
python scripts/db_backup.py --restore FILE --apply --tables users accounts   # some tables
```

Rows are upserted by primary key. Nothing is dropped, and rows that exist only
in the database are kept.

## Rebuild from nothing (cluster lost)

1. Create a new free TiDB Serverless cluster. Put its host, user and password in `.env`
   (`MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_SSL=true`).
2. Create the schema: `python -m backend.migrate` (creates the `flux` tables).
   If the `flux` database doesn't exist yet, run `python seed_mysql.py` instead,
   which creates it, runs the migrations and loads the reference data.
3. With a backup: restore it as above. Without one: `seed_mysql.py` reference data
   plus the demo user, which the API re-seeds on boot (`DEMO_RESEED`, backend/demo_seed.py).
   For price history: `python scripts/copy_history_to_tidb.py --apply`.
4. Create the app user with `scripts/db_app_user.sql`, then update `MYSQL_HOST`,
   `MYSQL_USER` and `MYSQL_PASSWORD` on Render.
