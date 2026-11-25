"""
tests/test_database.py  -  run with:  pytest tests/test_database.py -v
Requires the database to exist:       python -m database.setup_database
"""
import sqlite3

import numpy as np
import pandas as pd

from database.connection import get_read_only_connection
from database.generate_data import SEED, generate_customers, generate_products
from database.setup_database import clean_customers
from database.validate_data import run_quality_checks


def _query(sql: str) -> pd.DataFrame:
    conn = get_read_only_connection()
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


# ---------- data quality ----------
def test_all_quality_checks_pass():
    conn = get_read_only_connection()
    try:
        failed = [c for c in run_quality_checks(conn) if not c.passed]
    finally:
        conn.close()
    assert failed == [], [f"{c.name}: {c.detail}" for c in failed]


# ---------- safety ----------
def test_read_only_connection_blocks_writes():
    conn = get_read_only_connection()
    try:
        for stmt in ("DELETE FROM orders",
                     "INSERT INTO products VALUES (999,'x','x','x',1,1)",
                     "DROP TABLE customers"):
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                continue
            raise AssertionError(f"write was allowed: {stmt}")
    finally:
        conn.close()


# ---------- reproducibility and cleaning ----------
def test_generator_is_reproducible():
    a = generate_customers(np.random.default_rng(SEED))
    b = generate_customers(np.random.default_rng(SEED))
    pd.testing.assert_frame_equal(a, b)
    p1 = generate_products(np.random.default_rng(SEED))
    p2 = generate_products(np.random.default_rng(SEED))
    pd.testing.assert_frame_equal(p1, p2)


def test_clean_customers_removes_duplicates_and_fills_city():
    raw = pd.DataFrame({
        "customer_id": [1, 1, 2],
        "city": ["Delhi", "Delhi", None],
    })
    cleaned = clean_customers(raw)
    assert len(cleaned) == 2
    assert cleaned.loc[cleaned.customer_id == 2, "city"].iloc[0] == "Unknown"


# ---------- business definitions ----------
def test_vw_sales_only_contains_completed_orders():
    df = _query("""SELECT COUNT(*) AS n FROM vw_sales v
                   JOIN orders o ON o.order_id = v.order_id
                   WHERE o.status <> 'Completed'""")
    assert df["n"].iloc[0] == 0


def test_revenue_formula_matches_manual_calculation():
    view_total = _query("SELECT SUM(revenue) AS r FROM vw_sales")["r"].iloc[0]
    manual = _query("""SELECT SUM(i.quantity * i.unit_price * (1 - i.discount_pct)) AS r
                       FROM order_items i JOIN orders o USING (order_id)
                       WHERE o.status = 'Completed'""")["r"].iloc[0]
    assert abs(view_total - manual) < 1.0


# ---------- planted scenarios actually exist ----------
def test_march_2025_revenue_is_below_february():
    m = _query("SELECT order_month, SUM(revenue) AS r FROM vw_sales GROUP BY order_month")
    m = m.set_index("order_month")["r"]
    assert m["2025-03"] < m["2025-02"]


def test_electronics_declines_in_2025():
    e = _query("""SELECT order_month, SUM(revenue) AS r FROM vw_sales
                  WHERE category = 'Electronics' AND order_month IN ('2025-01','2025-06')
                  GROUP BY order_month""").set_index("order_month")["r"]
    assert e["2025-06"] < 0.7 * e["2025-01"]


def test_vip_reducers_cut_back():
    df = _query("""SELECT order_month, SUM(revenue) AS r FROM vw_sales
                   WHERE customer_id <= 8 GROUP BY order_month""").set_index("order_month")["r"]
    before = df.loc["2024-10":"2024-12"].mean()
    after = df.loc["2025-04":"2025-06"].mean()
    assert after < 0.3 * before


def test_central_is_lowest_region_and_festive_months_peak():
    reg = _query("SELECT order_region, SUM(revenue) AS r FROM vw_sales GROUP BY order_region")
    assert reg.sort_values("r").iloc[0]["order_region"] == "Central"
    orders = _query("""SELECT order_month, COUNT(DISTINCT order_id) AS n FROM vw_sales
                       WHERE order_month LIKE '2024-%' GROUP BY order_month"""
                    ).set_index("order_month")["n"]
    assert orders["2024-11"] > orders["2024-06"]