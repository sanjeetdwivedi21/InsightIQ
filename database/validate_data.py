"""
database/validate_data.py

Data-quality checks (run automatically by setup_database.py) and a scenario
report that shows the planted business patterns really exist in the data.

    python -m database.validate_data
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

import pandas as pd

from database.connection import get_read_only_connection

logger = logging.getLogger(__name__)

MIN_CUSTOMERS, MIN_PRODUCTS, MIN_ORDERS = 2000, 100, 10000


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def _scalar(conn: sqlite3.Connection, sql: str):
    return conn.execute(sql).fetchone()[0]


def run_quality_checks(conn: sqlite3.Connection) -> list[Check]:
    checks: list[Check] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append(Check(name, bool(passed), detail))

    n_cust = _scalar(conn, "SELECT COUNT(*) FROM customers")
    n_prod = _scalar(conn, "SELECT COUNT(*) FROM products")
    n_ord = _scalar(conn, "SELECT COUNT(*) FROM orders")
    add(f"At least {MIN_CUSTOMERS} customers", n_cust >= MIN_CUSTOMERS, f"found {n_cust}")
    add(f"At least {MIN_PRODUCTS} products", n_prod >= MIN_PRODUCTS, f"found {n_prod}")
    add(f"At least {MIN_ORDERS} orders", n_ord >= MIN_ORDERS, f"found {n_ord}")

    fk = conn.execute("PRAGMA foreign_key_check").fetchall()
    add("Foreign keys all valid (no orphan rows)", len(fk) == 0, f"{len(fk)} violations")

    dmin, dmax = conn.execute("SELECT MIN(order_date), MAX(order_date) FROM orders").fetchone()
    span_days = (pd.Timestamp(dmax) - pd.Timestamp(dmin)).days
    add("Orders span at least 2 years", span_days >= 700, f"{dmin} to {dmax}")

    statuses = {r[0] for r in conn.execute("SELECT DISTINCT status FROM orders")}
    add("All three order statuses present", statuses == {"Completed", "Cancelled", "Returned"}, str(sorted(statuses)))

    add("Multiple categories (>=5)", _scalar(conn, "SELECT COUNT(DISTINCT category) FROM products") >= 5)
    add("Multiple regions (>=4)", _scalar(conn, "SELECT COUNT(DISTINCT region) FROM customers") >= 4)
    add("Multiple payment methods (>=4)", _scalar(conn, "SELECT COUNT(DISTINCT payment_method) FROM payments") >= 4)
    add("Multiple customer segments (>=3)", _scalar(conn, "SELECT COUNT(DISTINCT segment) FROM customers") >= 3)

    bad_margin = _scalar(conn, "SELECT COUNT(*) FROM products WHERE unit_cost >= unit_price")
    add("Every product has cost below price", bad_margin == 0, f"{bad_margin} bad")

    no_items = _scalar(conn, "SELECT COUNT(*) FROM orders o WHERE NOT EXISTS "
                             "(SELECT 1 FROM order_items i WHERE i.order_id = o.order_id)")
    add("Every order has at least one item", no_items == 0, f"{no_items} without items")

    no_pay = _scalar(conn, "SELECT COUNT(*) FROM orders o WHERE NOT EXISTS "
                           "(SELECT 1 FROM payments p WHERE p.order_id = o.order_id)")
    add("Every order has a payment", no_pay == 0, f"{no_pay} without payment")

    mismatch = _scalar(conn, """
        SELECT COUNT(*) FROM (
          SELECT o.order_id,
                 SUM(i.quantity * i.unit_price * (1 - i.discount_pct)) AS total,
                 p.amount
          FROM orders o
          JOIN order_items i ON i.order_id = o.order_id
          JOIN payments p ON p.order_id = o.order_id
          GROUP BY o.order_id
          HAVING ABS(total - p.amount) > 0.05)""")
    add("Payment amount matches order total", mismatch == 0, f"{mismatch} mismatches")

    early = _scalar(conn, "SELECT COUNT(*) FROM orders o JOIN customers c USING (customer_id) "
                          "WHERE o.order_date < c.signup_date")
    add("No order placed before customer signup", early == 0, f"{early} violations")

    dup_email = _scalar(conn, "SELECT COUNT(*) - COUNT(DISTINCT email) FROM customers")
    add("No duplicate customer emails", dup_email == 0, f"{dup_email} duplicates")

    null_city = _scalar(conn, "SELECT COUNT(*) FROM customers WHERE city IS NULL")
    add("No NULL cities (filled with 'Unknown')", null_city == 0, f"{null_city} nulls")

    vw = _scalar(conn, "SELECT COUNT(*) FROM vw_sales")
    add("vw_sales view returns rows", vw > 0, f"{vw} rows")
    return checks


def scenario_report(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    """Numbers that prove the planted scenarios exist. All from the database."""
    q = lambda sql: pd.read_sql_query(sql, conn)  # noqa: E731
    return {
        "monthly": q("""
            SELECT order_month, ROUND(SUM(revenue)) AS revenue,
                   COUNT(DISTINCT order_id) AS orders,
                   ROUND(SUM(revenue) * 1.0 / COUNT(DISTINCT order_id)) AS aov
            FROM vw_sales GROUP BY order_month ORDER BY order_month"""),
        "electronics_vs_rest": q("""
            SELECT order_month,
                   ROUND(SUM(CASE WHEN category = 'Electronics' THEN revenue END)) AS electronics,
                   ROUND(SUM(CASE WHEN category <> 'Electronics' THEN revenue END)) AS other
            FROM vw_sales WHERE order_month BETWEEN '2025-01' AND '2025-06'
            GROUP BY order_month ORDER BY order_month"""),
        "vip_reducers": q("""
            SELECT order_month, ROUND(SUM(revenue)) AS revenue
            FROM vw_sales WHERE customer_id <= 8 AND order_month BETWEEN '2024-10' AND '2025-06'
            GROUP BY order_month ORDER BY order_month"""),
        "regions": q("""
            SELECT order_region, ROUND(SUM(revenue)) AS revenue
            FROM vw_sales GROUP BY order_region ORDER BY revenue DESC"""),
        "categories": q("""
            SELECT category, ROUND(SUM(revenue)) AS revenue, ROUND(SUM(profit)) AS profit
            FROM vw_sales GROUP BY category ORDER BY revenue DESC"""),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    conn = get_read_only_connection()
    try:
        results = run_quality_checks(conn)
        for c in results:
            print(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}  {c.detail}")
        failed = [c for c in results if not c.passed]
        print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
        print("\n--- Scenario report ---")
        for name, df in scenario_report(conn).items():
            print(f"\n{name}:\n{df.to_string(index=False)}")
        return 1 if failed else 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())