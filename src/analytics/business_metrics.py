"""
src/analytics/business_metrics.py

Core KPI definitions (the same ones the SQL view uses):
  revenue = Completed orders, after discount     profit = revenue - cost
  AOV = revenue / number of Completed orders
"""
from __future__ import annotations

import pandas as pd

from src.analytics.common import require_columns, safe_div

_SALES = ["order_id", "customer_id", "order_month", "quantity", "revenue", "profit", "gross_revenue"]
_ORDERS = ["order_id", "order_month", "status"]


def _core(sales: pd.DataFrame) -> dict:
    revenue = float(sales["revenue"].sum())
    profit = float(sales["profit"].sum())
    gross = float(sales["gross_revenue"].sum())
    orders = int(sales["order_id"].nunique())
    lines = int(len(sales))
    return {
        "revenue": revenue,
        "profit": profit,
        "margin_pct": None if revenue == 0 else profit / revenue * 100,
        "orders": orders,
        "aov": safe_div(revenue, orders),
        "units": int(sales["quantity"].sum()),
        "line_items": lines,
        "items_per_order": safe_div(lines, orders),
        "revenue_per_item": safe_div(revenue, lines),
        "discount_pct": None if gross == 0 else (gross - revenue) / gross * 100,
        "customers": int(sales["customer_id"].nunique()),
    }


def _status_rates(orders: pd.DataFrame) -> dict:
    total = len(orders)
    cancelled = int((orders["status"] == "Cancelled").sum())
    returned = int((orders["status"] == "Returned").sum())
    return {
        "orders_all": total, "cancelled_orders": cancelled, "returned_orders": returned,
        "cancel_rate_pct": None if total == 0 else cancelled / total * 100,
        "return_rate_pct": None if total == 0 else returned / total * 100,
    }


def period_metrics(sales: pd.DataFrame, orders: pd.DataFrame, month: str) -> dict:
    """All KPIs for one month (YYYY-MM)."""
    require_columns(sales, _SALES, "sales")
    require_columns(orders, _ORDERS, "orders")
    out = _core(sales[sales["order_month"] == month])
    out.update(_status_rates(orders[orders["order_month"] == month]))
    out["month"] = month
    return out


def kpi_summary(sales: pd.DataFrame, orders: pd.DataFrame) -> dict:
    """All-time headline KPIs (used by the dashboard in a later phase)."""
    require_columns(sales, _SALES, "sales")
    require_columns(orders, _ORDERS, "orders")
    out = _core(sales)
    out.update(_status_rates(orders))
    return out