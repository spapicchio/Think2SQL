#!/usr/bin/env python3
"""
Merge all prediction cache SQLite files into:
  .think2sql_cache/pred_cached_query.sqlite

Assumptions:
- Each source DB has a single table `cache_data`.
- Exclude `target_cached_query.sqlite` and the destination itself.
- Append rows with INSERT OR IGNORE (dedup via your unique key, e.g., hash_key).
"""

import sqlite3
import sys
from pathlib import Path

import fire


# ---------- small helpers ----------

def list_source_dbs(root: Path, excluded_names) -> list[Path]:
    """Find all .sqlite files under ROOT, excluding destination and target cache.
    :param excluded_names:
    """
    return sorted(
        p for p in root.rglob("*.sqlite")
        if p.is_file() and p.name not in excluded_names
    )


def table_exists(conn: sqlite3.Connection, table: str, schema: str = "main") -> bool:
    cur = conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type='table' AND name=?;", (table,)
    )
    row = cur.fetchone()
    cur.close()
    return row is not None


def get_create_table_sql(conn: sqlite3.Connection, table: str, schema: str = "main") -> str | None:
    """Return CREATE TABLE statement for schema.table, or None."""
    cur = conn.execute(
        f"SELECT sql FROM {schema}.sqlite_master WHERE type='table' AND name=?;", (table,)
    )
    row = cur.fetchone()
    cur.close()
    return row[0] if row else None


def get_columns(conn: sqlite3.Connection, table: str, schema: str = "main") -> list[str]:
    """Return ordered column names for schema.table."""
    cur = conn.execute(f'PRAGMA {schema}.table_info("{table}");')
    cols = [r[1] for r in cur.fetchall()]
    cur.close()
    return cols


def ensure_cache_table_from_src(dest: sqlite3.Connection) -> None:
    """
    Create main.cache_data if missing by copying the CREATE from src.cache_data.
    NOTE: This requires the source DB to be ATTACHED as 'src'.
    """
    if table_exists(dest, "cache_data", "main"):
        return
    create_sql = get_create_table_sql(dest, "cache_data", "src")
    if not create_sql:
        raise RuntimeError("Source DB lacks `cache_data` or CREATE SQL not found.")
    # Make idempotent on first creation
    create_sql = create_sql.replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ", 1)
    dest.execute(create_sql)


def merge_one_source(dest: sqlite3.Connection, src_path: Path) -> int:
    """
    ATTACH source as 'src', copy schema if needed, then INSERT OR IGNORE
    the intersection of columns from src.cache_data into main.cache_data.
    Returns inserted row count (best effort).
    """
    # ATTACH outside of any open statement/iterator
    dest.execute("ATTACH DATABASE ? AS src;", (str(src_path),))

    try:
        # Read schema info BEFORE starting the write txn, and close cursors promptly
        main_has_table = table_exists(dest, "cache_data", "main")
        if not main_has_table:
            # We need the CREATE statement from the src
            create_sql = get_create_table_sql(dest, "cache_data", "src")
            if not create_sql:
                # Nothing to copy from; treat as empty
                return 0

        # Compute common columns (ordered as in MAIN when available)
        if not main_has_table:
            # If main doesn't exist yet, create it first so we can read its columns
            dest.execute(create_sql.replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ", 1))
        main_cols = get_columns(dest, "cache_data", "main")
        src_cols = set(get_columns(dest, "cache_data", "src"))
        common_cols = [c for c in main_cols if c in src_cols]
        if not common_cols:
            return 0

        cols_csv = ", ".join(f'"{c}"' for c in common_cols)

        # Short, per-source transaction:
        #  - guarantees atomic insert from this source
        #  - ensures all statements finish before DETACH
        with dest:
            # If main table still somehow missing (race), ensure again via src
            ensure_cache_table_from_src(dest)

            # INSERT OR IGNORE to avoid duplicate key errors (e.g., same hash_key)
            cur = dest.execute(
                f'INSERT OR IGNORE INTO "cache_data" ({cols_csv}) '
                f'SELECT {cols_csv} FROM src."cache_data";'
            )
            inserted = cur.rowcount if cur.rowcount is not None else 0
            cur.close()

        # Transaction has been committed here; safe to detach now.
        return inserted

    finally:
        # Very important: DETACH only after all cursors are closed and txn committed.
        # Wrap in a tiny retry just in case the connection needs a moment.
        try:
            dest.execute("DETACH DATABASE src;")
        except sqlite3.OperationalError as e:
            # Last resort: finalize any pending work and try once more.
            dest.commit()
            dest.execute("DETACH DATABASE src;")


# ---------- main ----------

def main(root, dest, excluded_names, remove_old_folders: bool = True):
    root = Path(root)
    excluded_names = set(excluded_names.split())
    excluded_names.add(dest)
    dest = root / dest
    if not root.exists():
        print(f"Directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    sources = list_source_dbs(root, excluded_names)
    if not sources:
        print("No source .sqlite files found. Nothing to merge.")
        return

    print(f"Found {len(sources)} source DBs.")
    print(f"Destination: {dest}")

    conn = sqlite3.connect(str(dest))
    try:
        # Pragmas: make writes faster and reduce spurious FK/lock issues
        conn.execute("PRAGMA busy_timeout=5000;")  # wait a bit if file is busy
        conn.execute("PRAGMA journal_mode=DELETE;")  # simple journal is fine for one-writer
        conn.execute("PRAGMA synchronous=OFF;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA foreign_keys=OFF;")

        total = 0
        for src in sources:
            print(f"Merging: {src}")
            inserted = merge_one_source(conn, src)
            total += max(inserted, 0)
            if remove_old_folders:
                Path(src).unlink()
                Path(src).parent.rmdir()

        print(f"Done. Rows inserted (best effort): {total}")
        print(f"Combined DB at: {dest}")

        # Optional: compact the file
        # conn.execute("VACUUM;")

        # Optional: enforce dedup at DB level
        # with conn:
        #     conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS ux_cache_data_hash_key ON cache_data(hash_key);')

    finally:
        conn.close()


if __name__ == "__main__":
    # ROOT = Path(".think2sql_cache").resolve()
    # DEST = ROOT / "pred_cached_query.sqlite"
    # EXCLUDE_NAMES = {"target_cached_query.sqlite", DEST.name}

    fire.Fire(main)
