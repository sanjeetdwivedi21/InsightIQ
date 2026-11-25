"""
database/connection.py

Database connections.
  - get_read_only_connection(): used by anything that runs queries (including,
    later, LLM-generated SQL). SQLite refuses writes on this connection.
  - get_write_connection(): used ONLY by setup_database.py to build the database.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from config.settings import get_settings


def _db_path(path: Path | None = None) -> Path:
    return Path(path) if path else Path(get_settings().business_db_path)


def get_read_only_connection(path: Path | None = None) -> sqlite3.Connection:
    db_path = _db_path(path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. Run: python -m database.setup_database"
        )
    # mode=ro makes SQLite itself reject INSERT/UPDATE/DELETE/DROP.
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def get_write_connection(path: Path | None = None) -> sqlite3.Connection:
    db_path = _db_path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn