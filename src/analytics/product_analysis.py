"""


Category and product analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.common import require_columns, shift_month
from src.analytics.revenue_analysis import group_change


def category_performance(sales: pd.DataFrame) -> pd.DataFrame:
    require_columns(sales, ["category", "order_id", "quantity", "revenue", "profit"], "sales")
    g = sales.groupby("category").agg(revenue=("revenue", "sum"), profit=("profit", "sum"),
                                      units=("quantity", "sum"), orders=("order_id", "nunique")).reset_index()
    g["margin_pct"] = g["profit"] / g["revenue"].replace(0, np.nan) * 100
    g["revenue_share_pct"] = g["revenue"] / g["revenue"].sum() * 100
    g["profit_share_pct"] = g["profit"] / g["profit"].sum() * 100
    return g.sort_values("revenue", ascending=False).reset_index(drop=True)


def category_monthly_revenue(sales: pd.DataFrame) -> pd.DataFrame:
    """Rows = months (every month in range), columns = categories, values = revenue."""
    require_columns(sales, ["category", "order_month", "revenue"], "sales")
    if sales.empty:
        return pd.DataFrame()
    pv = sales.pivot_table(index="order_month", columns="category", values="revenue", aggfunc="sum", fill_value=0.0)
    full = pd.period_range(pv.index.min(), pv.index.max(), freq="M").astype(str)
    return pv.reindex(full, fill_value=0.0).rename_axis("month")


def category_change(sales: pd.DataFrame, month: str, prev: str) -> pd.DataFrame:
    return group_change(sales, month, prev, "category")


def decline_streak(series: pd.Series, month: str) -> int:
    """How many consecutive months, ending at `month`, revenue fell versus the month before."""
    s = series.sort_index()
    if month not in s.index:
        return 0
    pos = s.index.get_loc(month)
    n = 0
    while pos - n - 1 >= 0 and s.iloc[pos - n] < s.iloc[pos - n - 1]:
        n += 1
    return n


def vs_prior_average_pct(series: pd.Series, month: str, window: int = 3):
    """Revenue in `month` versus the average of the `window` months before it (percent), or None."""
    prior = [m for m in (shift_month(month, -i) for i in range(1, window + 1)) if m in series.index]
    if not prior or month not in series.index:
        return None
    avg = float(series.loc[prior].mean())
    return None if avg == 0 else (float(series.loc[month]) - avg) / avg * 100


def top_products(sales: pd.DataFrame, n: int = 10, by: str = "revenue") -> pd.DataFrame:
    require_columns(sales, ["product_id", "product_name", "category", "quantity", "revenue", "profit"], "sales")
    if by not in ("revenue", "profit", "units"):
        raise ValueError("by must be 'revenue', 'profit' or 'units'")
    g = sales.groupby(["product_id", "product_name", "category"]).agg(
        revenue=("revenue", "sum"), profit=("profit", "sum"), units=("quantity", "sum")).reset_index()
    return g.sort_values(by, ascending=False).head(n).reset_index(drop=True)