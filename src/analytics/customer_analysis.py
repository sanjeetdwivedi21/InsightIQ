"""


Customer-level analysis: who buys, how often, how concentrated revenue is, who changed
their buying between two months, and whether high-value customers pulled back.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.analytics.common import require_columns, safe_div, shift_month


def customer_summary(sales: pd.DataFrame) -> pd.DataFrame:
    require_columns(sales, ["customer_id", "order_id", "order_date", "quantity", "revenue", "profit"], "sales")
    g = sales.groupby("customer_id").agg(revenue=("revenue", "sum"), profit=("profit", "sum"),
                                         orders=("order_id", "nunique"), units=("quantity", "sum"),
                                         first_order=("order_date", "min"), last_order=("order_date", "max")).reset_index()
    g["aov"] = g["revenue"] / g["orders"].replace(0, np.nan)
    g["share_pct"] = g["revenue"] / g["revenue"].sum() * 100
    return g.sort_values("revenue", ascending=False).reset_index(drop=True)


def top_customers(sales: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    top = customer_summary(sales).head(n).copy()
    top["cumulative_share_pct"] = top["share_pct"].cumsum()
    return top


def segment_performance(sales: pd.DataFrame) -> pd.DataFrame:
    require_columns(sales, ["segment", "customer_id", "order_id", "revenue", "profit"], "sales")
    g = sales.groupby("segment").agg(revenue=("revenue", "sum"), profit=("profit", "sum"),
                                     orders=("order_id", "nunique"), customers=("customer_id", "nunique")).reset_index()
    g["aov"] = g["revenue"] / g["orders"].replace(0, np.nan)
    g["revenue_per_customer"] = g["revenue"] / g["customers"].replace(0, np.nan)
    g["share_pct"] = g["revenue"] / g["revenue"].sum() * 100
    return g.sort_values("revenue", ascending=False).reset_index(drop=True)


def customer_change_buckets(sales: pd.DataFrame, month: str, prev: str):
    """Split every customer who bought in either month into new / lost / reduced / increased / flat.
    The revenue effects of the buckets add up exactly to the net revenue change.
    Returns (per-customer table sorted by biggest drop first, summary dict)."""
    require_columns(sales, ["customer_id", "order_month", "revenue"], "sales")
    pair = sales[sales["order_month"].isin([prev, month])]
    empty_summary = {k: {"customers": 0, "revenue_effect": 0.0, "share_of_net_change_pct": None}
                     for k in ("new", "lost", "reduced", "increased", "flat")}
    if pair.empty:
        return pd.DataFrame(columns=["customer_id", "prev_revenue", "revenue", "change", "status"]), empty_summary
    pv = pair.pivot_table(index="customer_id", columns="order_month", values="revenue", aggfunc="sum", fill_value=0.0)
    pv = pv.reindex(columns=[prev, month], fill_value=0.0)
    t = pd.DataFrame({"customer_id": pv.index, "prev_revenue": pv[prev].to_numpy(), "revenue": pv[month].to_numpy()})
    t["change"] = t["revenue"] - t["prev_revenue"]
    conditions = [
        (t["prev_revenue"] == 0) & (t["revenue"] > 0),
        (t["prev_revenue"] > 0) & (t["revenue"] == 0),
        (t["prev_revenue"] > 0) & (t["revenue"] > 0) & (t["change"] < 0),
        (t["prev_revenue"] > 0) & (t["revenue"] > 0) & (t["change"] > 0),
    ]
    t["status"] = np.select(conditions, ["new", "lost", "reduced", "increased"], default="flat")
    net = float(t["change"].sum())
    summary = {}
    for k in ("new", "lost", "reduced", "increased", "flat"):
        sub = t[t["status"] == k]
        effect = float(sub["change"].sum())
        summary[k] = {"customers": int(len(sub)), "revenue_effect": effect,
                      "share_of_net_change_pct": None if net == 0 else effect / net * 100}
    return t.sort_values("change").reset_index(drop=True), summary


def high_value_change(sales: pd.DataFrame, month: str, prev: str, window: int = 6,
                      top_pct: float = 0.05) -> pd.DataFrame:
    """Did the biggest customers pull back? 'High-value' = top `top_pct` of customers by revenue
    over the `window` months ending at `prev`. Each is compared with their OWN usual monthly
    spend (baseline average over that window), which is fairer than comparing two single months.
    Sorted by biggest drop first."""
    require_columns(sales, ["customer_id", "order_month", "revenue"], "sales")
    cols = ["customer_id", "baseline_total", "baseline_monthly_avg", "revenue", "change_vs_baseline",
            "pct_vs_baseline", "sharply_reduced"]
    if sales.empty:
        return pd.DataFrame(columns=cols)
    first = sales["order_month"].min()
    start = max(shift_month(prev, -(window - 1)), first)
    win = list(pd.period_range(start, prev, freq="M").astype(str))
    base = sales[sales["order_month"].isin(win)].groupby("customer_id")["revenue"].sum()
    if base.empty:
        return pd.DataFrame(columns=cols)
    k = max(1, int(math.ceil(top_pct * len(base))))
    top = base.sort_values(ascending=False).head(k)
    cur = sales[sales["order_month"] == month].groupby("customer_id")["revenue"].sum()
    df = pd.DataFrame({"customer_id": top.index, "baseline_total": top.to_numpy()})
    df["baseline_monthly_avg"] = df["baseline_total"] / len(win)
    df["revenue"] = df["customer_id"].map(cur).fillna(0.0)
    df["change_vs_baseline"] = df["revenue"] - df["baseline_monthly_avg"]
    df["pct_vs_baseline"] = df["change_vs_baseline"] / df["baseline_monthly_avg"].replace(0, np.nan) * 100
    df["sharply_reduced"] = df["revenue"] < 0.5 * df["baseline_monthly_avg"]
    return df.sort_values("change_vs_baseline").reset_index(drop=True)[cols]


def high_value_calibration(sales: pd.DataFrame, month: str, window: int = 6, top_pct: float = 0.05,
                           min_history: int = 4) -> dict | None:
    """Put the high-value-customer result for `month` in context.

    Top customers buy in lumps, and a group picked for high PAST spend tends to spend less than its
    own average the next month even when nothing is wrong (regression to the mean). So the result for
    one month is compared with the same measure in EARLIER months:
      - cohort_pct: combined spend of the cohort vs its combined usual monthly spend
      - sharp_share_pct: share of cohort members who spent under half their usual amount
    Returns None when there are fewer than `min_history` earlier months with a full baseline window."""
    require_columns(sales, ["customer_id", "order_month", "revenue"], "sales")
    months = sorted(sales["order_month"].unique())
    if month not in months:
        return None

    def measure(m: str):
        t = high_value_change(sales, m, shift_month(m, -1), window=window, top_pct=top_pct)
        if t.empty or t["baseline_monthly_avg"].sum() == 0:
            return None
        return (float(t["revenue"].sum() / t["baseline_monthly_avg"].sum() - 1) * 100,
                float(t["sharply_reduced"].mean() * 100))

    usable = [m for i, m in enumerate(months) if i >= window and m <= month]
    results = {m: measure(m) for m in usable}
    results = {m: v for m, v in results.items() if v is not None}
    if month not in results:
        return None
    history = [v for m, v in results.items() if m < month]
    if len(history) < min_history:
        return None
    cohort_hist = np.array([h[0] for h in history])
    share_hist = np.array([h[1] for h in history])
    typical = float(cohort_hist.mean())
    cohort_std = float(cohort_hist.std(ddof=1)) if len(cohort_hist) > 1 else None
    share_std = float(share_hist.std(ddof=1)) if len(share_hist) > 1 else None
    this_pct, this_share = results[month]
    # consecutive months (ending at `month`) in which the cohort was below its typical level
    streak = 0
    for m in sorted(results, reverse=True):
        if results[m][0] < typical:
            streak += 1
        else:
            break
    return {
        "history_months": len(history),
        "cohort_pct": this_pct,
        "typical_cohort_pct": typical,
        "cohort_pct_std": cohort_std,
        "cohort_z": None if not cohort_std else (this_pct - typical) / cohort_std,
        "sharp_share_pct": this_share,
        "typical_sharp_share_pct": float(share_hist.mean()),
        "sharp_share_std": share_std,
        "months_below_typical_in_a_row": streak,
    }


def customer_behavior(sales: pd.DataFrame) -> dict:
    """Purchasing-behaviour summary: repeat rate, orders per customer, time between orders,
    recency and how concentrated revenue is."""
    require_columns(sales, ["customer_id", "order_id", "order_date", "order_month", "revenue"], "sales")
    if sales.empty:
        return {}
    cs = customer_summary(sales)
    n = len(cs)
    orders = sales.drop_duplicates("order_id")[["customer_id", "order_id", "order_date"]].sort_values("order_date")
    gaps = orders.groupby("customer_id")["order_date"].diff().dt.days.dropna()
    cum = cs["revenue"].cumsum() / cs["revenue"].sum()
    top10 = max(1, int(math.ceil(0.10 * n)))
    last_month = sales["order_month"].max()
    recent_months = {shift_month(last_month, -i) for i in range(3)}
    active_recent = sales[sales["order_month"].isin(recent_months)]["customer_id"].nunique()
    return {
        "customers": int(n),
        "avg_orders_per_customer": float(cs["orders"].mean()),
        "avg_revenue_per_customer": float(cs["revenue"].mean()),
        "repeat_customer_rate_pct": float((cs["orders"] >= 2).mean() * 100),
        "avg_days_between_orders": None if gaps.empty else float(gaps.mean()),
        "median_days_between_orders": None if gaps.empty else float(gaps.median()),
        "top_10pct_customers_revenue_share_pct": float(cs["revenue"].head(top10).sum() / cs["revenue"].sum() * 100),
        "customers_for_80pct_of_revenue": int((cum < 0.8).sum() + 1),
        "active_last_3_months_pct": active_recent / n * 100,
        "last_data_month": last_month,
    }