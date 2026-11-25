"""
database/setup_database.py

Reads the raw CSVs, cleans them, builds the SQLite database from schema.sql,
and runs data-quality checks.

    python -m database.setup_database
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from database.connection import get_read_only_connection, get_write_connection, _db_path
from database.validate_data import run_quality_checks

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
SCHEMA_FILE = BASE_DIR / "database" / "schema.sql"

# Load order matters: parents before children (foreign keys).
TABLES = ["customers", "products", "orders", "order_items", "payments"]

logger = logging.getLogger(__name__)


def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset="customer_id", keep="first")
    logger.info("customers: removed %d duplicate rows", before - len(df))
    missing = int(df["city"].isna().sum())
    df = df.assign(city=df["city"].fillna("Unknown"))
    logger.info("customers: filled %d missing cities with 'Unknown'", missing)
    return df


def load_raw() -> dict[str, pd.DataFrame]:
    data = {}
    for t in TABLES:
        f = RAW_DIR / f"{t}.csv"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing. Run: python -m database.generate_data")
        data[t] = pd.read_csv(f)
    data["customers"] = clean_customers(data["customers"])
    return data


def build_database() -> Path:
    data = load_raw()
    db_path = _db_path()
    if db_path.exists():
        db_path.unlink()          # rebuild from scratch so the script is repeatable

    conn = get_write_connection()
    try:
        conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
        for t in TABLES:
            data[t].to_sql(t, conn, if_exists="append", index=False)
            logger.info("loaded %-12s %6d rows", t, len(data[t]))
        conn.commit()
    finally:
        conn.close()
    return db_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    db_path = build_database()
    logger.info("\nDatabase created: %s\n\nRunning data-quality checks...", db_path)

    conn = get_read_only_connection()
    try:
        results = run_quality_checks(conn)
    finally:
        conn.close()
    for c in results:
        print(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}  {c.detail}")
    failed = [c for c in results if not c.passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())