"""
src/sql/schema_provider.py

Reads the live database schema and turns it into compact text for the LLM prompt:
tables, columns, keys, the exact values of categorical columns, and the date range.
Giving the model the real values ('Completed', 'North', ...) prevents most wrong filters.
"""
from __future__ import annotations

from functools import lru_cache

from database.connection import get_read_only_connection
from src.sql.sql_validator import ALLOWED_TABLES

# columns whose distinct values are shown to the model
CATEGORICAL_COLUMNS = {
    "customers": ["segment", "region"],
    "products": ["category"],
    "orders": ["status", "region"],
    "payments": ["payment_method", "payment_status"],
}
MAX_VALUES = 15
TABLE_ORDER = ["customers", "products", "orders", "order_items", "payments", "vw_sales"]


def build_schema_text(conn) -> str:
    lines: list[str] = []
    for table in TABLE_ORDER:
        if table not in ALLOWED_TABLES:
            continue
        cols = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        if not cols:
            continue
        kind = "VIEW" if table == "vw_sales" else "TABLE"
        col_text = ", ".join(f"{c[1]} {c[2] or ''}{' PK' if c[5] else ''}".strip() for c in cols)
        lines.append(f"{kind} {table} ({col_text})")
        for fk in conn.execute(f"PRAGMA foreign_key_list('{table}')").fetchall():
            lines.append(f"  FOREIGN KEY {table}.{fk[3]} -> {fk[2]}.{fk[4]}")
        for col in CATEGORICAL_COLUMNS.get(table, []):
            vals = [r[0] for r in conn.execute(
                f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL ORDER BY 1 LIMIT {MAX_VALUES}')]
            lines.append(f"  values of {table}.{col}: " + " | ".join(str(v) for v in vals))
        lines.append("")
    dmin, dmax = conn.execute("SELECT MIN(order_date), MAX(order_date) FROM orders").fetchone()
    lines.append(f"DATA RANGE: orders.order_date runs from {dmin} to {dmax}.")
    return "\n".join(lines).strip()


@lru_cache(maxsize=1)
def get_schema_text() -> str:
    conn = get_read_only_connection()
    try:
        return build_schema_text(conn)
    finally:
        conn.close()