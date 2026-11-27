"""

Loads the two DataFrames every analysis starts from (read-only connection):
  sales  - one row per sold line item, COMPLETED orders only (the vw_sales view)
  orders - one row per order, ALL statuses (for cancellation / return rates)
The analytics functions themselves only receive DataFrames, so they can be
tested with tiny hand-made data.
"""
from __future__ import annotations

import pandas as pd

from database.connection import get_read_only_connection

SALES_COLUMNS = ["order_id", "order_date", "order_month", "order_region", "customer_id", "segment",
                 "product_id", "product_name", "category", "quantity", "revenue", "cost", "profit"]
ORDER_COLUMNS = ["order_id", "customer_id", "order_date", "order_month", "status", "region"]


def load_sales(conn=None) -> pd.DataFrame:
    own = conn is None
    conn = conn or get_read_only_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM vw_sales", conn)
    finally:
        if own:
            conn.close()
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["gross_revenue"] = df["quantity"] * df["unit_price"]          # before discount
    return df


def load_orders(conn=None) -> pd.DataFrame:
    own = conn is None
    conn = conn or get_read_only_connection()
    try:
        df = pd.read_sql_query(
            "SELECT order_id, customer_id, order_date, status, region, "
            "strftime('%Y-%m', order_date) AS order_month FROM orders", conn)
    finally:
        if own:
            conn.close()
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df