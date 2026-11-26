"""
src/sql/query_executor.py

Runs validated SQL on the READ-ONLY database connection with:
  - a time limit (query is interrupted if it runs too long)
  - a row limit (extra rows are dropped and `truncated` is set)

The executor re-validates the SQL itself, so it is safe even if a caller forgets to.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pandas as pd

from config.settings import get_settings
from database.connection import get_read_only_connection
from src.models import SQLResult
from src.sql.sql_validator import validate_sql

PROGRESS_STEPS = 10_000   # how often SQLite checks the clock (in VM instructions)


class QueryExecutionError(Exception):
    """The database rejected the query (bad column, bad syntax, ...)."""


class QueryTimeoutError(Exception):
    """The query ran longer than the allowed time and was stopped."""


def execute_query(sql: str, row_limit: int | None = None, timeout_seconds: float | None = None,
                  db_path: Path | None = None) -> SQLResult:
    settings = get_settings()
    row_limit = row_limit or settings.sql_row_limit
    timeout = timeout_seconds or settings.sql_timeout_seconds

    cleaned = validate_sql(sql)                      # raises SQLValidationError

    conn = get_read_only_connection(db_path)
    deadline = time.monotonic() + timeout
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, PROGRESS_STEPS)
    start = time.monotonic()
    try:
        cur = conn.execute(cleaned)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(row_limit + 1)
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise QueryTimeoutError(f"The query took longer than {timeout} seconds and was stopped.") from None
        raise QueryExecutionError(str(exc)) from None
    except sqlite3.Error as exc:
        raise QueryExecutionError(str(exc)) from None
    finally:
        conn.close()

    truncated = len(rows) > row_limit
    rows = rows[:row_limit]
    df = pd.DataFrame.from_records(rows, columns=columns)
    return SQLResult(sql=cleaned, dataframe=df, row_count=len(df),
                     elapsed_ms=(time.monotonic() - start) * 1000, truncated=truncated)