"""

Time-based and group-based revenue analysis (pure Pandas/NumPy, DataFrame in -> result out):
monthly trend with month-over-month and year-over-year change, growth/decline summary,
regional analysis, a generic "what changed between two months, by group" table, and the
order-volume vs average-order-value split of a revenue change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.common import AnalyticsError, pct_change, require_columns, safe_div

_COLS = ["order_id", "order_month", "customer_id", "quantity", "revenue", "profit"]


def monthly_summary(sales: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar month (months without sales appear with zeros), with
    revenue, profit, margin, orders, AOV, units, customers, MoM and YoY change."""
    require_columns(sales, _COLS, "sales")
    if sales.empty:
        return pd.DataFrame(columns=["month", "revenue", "profit", "orders", "units", "customers", "aov",
                                     "margin_pct", "mom_change_abs", "mom_change_pct", "yoy_change_pct"])
    g = sales.groupby("order_month").agg(revenue=("revenue", "sum"), profit=("profit", "sum"),
                                         orders=("order_id", "nunique"), units=("quantity", "sum"),
                                         customers=("customer_id", "nunique"))
    full = pd.period_range(g.index.min(), g.index.max(), freq="M").astype(str)
    g = g.reindex(full, fill_value=0)
    g.index.name = "month"
    g["aov"] = (g["revenue"] / g["orders"].replace(0, np.nan))
    g["margin_pct"] = (g["profit"] / g["revenue"].replace(0, np.nan)) * 100
    g["mom_change_abs"] = g["revenue"].diff()
    g["mom_change_pct"] = (g["revenue"] / g["revenue"].shift(1).replace(0, np.nan) - 1) * 100
    g["yoy_change_pct"] = (g["revenue"] / g["revenue"].shift(12).replace(0, np.nan) - 1) * 100
    return g.reset_index()


def growth_decline_analysis(monthly: pd.DataFrame) -> dict:
    """Summarise the trend: best/worst months, biggest swings, linear trend, volatility, yearly totals."""
    if monthly.empty:
        return {}
    m = monthly.copy()
    best = m.loc[m["revenue"].idxmax()]
    worst = m.loc[m["revenue"].idxmin()]
    mom = m.dropna(subset=["mom_change_pct"])
    drop = mom.loc[mom["mom_change_pct"].idxmin()] if not mom.empty else None
    rise = mom.loc[mom["mom_change_pct"].idxmax()] if not mom.empty else None
    slope = float(np.polyfit(np.arange(len(m)), m["revenue"].to_numpy(dtype=float), 1)[0]) if len(m) >= 2 else 0.0
    mean_rev = float(m["revenue"].mean())
    years = m.assign(year=m["month"].str[:4]).groupby("year")["revenue"].sum()
    year_rows = {y: float(v) for y, v in years.items()}
    yoy = {}
    ys = sorted(year_rows)
    for a, b in zip(ys, ys[1:]):
        yoy[f"{a}->{b}"] = pct_change(year_rows[b], year_rows[a])
    return {
        "months": int(len(m)),
        "best_month": {"month": best["month"], "revenue": float(best["revenue"])},
        "worst_month": {"month": worst["month"], "revenue": float(worst["revenue"])},
        "largest_mom_drop": None if drop is None else {"month": drop["month"], "pct": float(drop["mom_change_pct"]),
                                                      "abs": float(drop["mom_change_abs"])},
        "largest_mom_rise": None if rise is None else {"month": rise["month"], "pct": float(rise["mom_change_pct"]),
                                                      "abs": float(rise["mom_change_abs"])},
        "declining_months": int((mom["mom_change_pct"] < 0).sum()),
        "rising_months": int((mom["mom_change_pct"] > 0).sum()),
        "trend_slope_per_month": slope,
        "trend_pct_of_mean_per_month": None if mean_rev == 0 else slope / mean_rev * 100,
        "mom_volatility_std_pct": None if len(mom) < 2 else float(mom["mom_change_pct"].std(ddof=1)),
        "yearly_revenue": year_rows,
        "yearly_change_pct": yoy,
    }


def group_change(sales: pd.DataFrame, month: str, prev: str, by: str) -> pd.DataFrame:
    """Revenue by `by` (category, region, segment, ...) in two months, with each group's change and
    its share of the net change. Sorted so the biggest decreases come first.

    contribution_pct         = group change / net change * 100 (sums to 100; unstable if net change ~ 0)
    share_of_decreases_pct   = group change / sum of all decreases * 100 (decreasing groups only)
    share_of_increases_pct   = group change / sum of all increases * 100 (increasing groups only)
    """
    require_columns(sales, [by, "order_month", "revenue"], "sales")
    cols = [by, "prev_revenue", "revenue", "change", "pct_change", "contribution_pct",
            "share_of_decreases_pct", "share_of_increases_pct", "prev_share_pct"]
    pair = sales[sales["order_month"].isin([prev, month])]
    if pair.empty:
        return pd.DataFrame(columns=cols)
    pv = pair.pivot_table(index=by, columns="order_month", values="revenue", aggfunc="sum", fill_value=0.0)
    pv = pv.reindex(columns=[prev, month], fill_value=0.0)
    out = pd.DataFrame({by: pv.index, "prev_revenue": pv[prev].to_numpy(), "revenue": pv[month].to_numpy()})
    out["change"] = out["revenue"] - out["prev_revenue"]
    out["pct_change"] = np.where(out["prev_revenue"] > 0, out["change"] / out["prev_revenue"].replace(0, np.nan) * 100, np.nan)
    net = out["change"].sum()
    dec = out.loc[out["change"] < 0, "change"].sum()
    inc = out.loc[out["change"] > 0, "change"].sum()
    out["contribution_pct"] = out["change"] / net * 100 if net != 0 else np.nan
    out["share_of_decreases_pct"] = np.where(out["change"] < 0, out["change"] / dec * 100 if dec != 0 else np.nan, 0.0)
    out["share_of_increases_pct"] = np.where(out["change"] > 0, out["change"] / inc * 100 if inc != 0 else np.nan, 0.0)
    tot_prev = out["prev_revenue"].sum()
    out["prev_share_pct"] = out["prev_revenue"] / tot_prev * 100 if tot_prev else np.nan
    return out.sort_values("change").reset_index(drop=True)[cols]


def region_performance(sales: pd.DataFrame) -> pd.DataFrame:
    require_columns(sales, ["order_region", "order_id", "revenue", "profit"], "sales")
    g = sales.groupby("order_region").agg(revenue=("revenue", "sum"), profit=("profit", "sum"),
                                          orders=("order_id", "nunique")).reset_index()
    g["aov"] = g["revenue"] / g["orders"].replace(0, np.nan)
    g["margin_pct"] = g["profit"] / g["revenue"].replace(0, np.nan) * 100
    g["share_pct"] = g["revenue"] / g["revenue"].sum() * 100
    return g.sort_values("revenue", ascending=False).reset_index(drop=True)


def region_change(sales: pd.DataFrame, month: str, prev: str) -> pd.DataFrame:
    return group_change(sales, month, prev, "order_region")


def _share(part, total):
    ratio = safe_div(part, total)
    return None if ratio is None else ratio * 100


def decompose_revenue_change(orders0, aov0, orders1, aov1) -> dict:
    """Exact arithmetic split of the change in revenue (= orders x AOV):
         volume_effect = (orders1 - orders0) * aov0     (more/fewer orders at the old AOV)
         aov_effect    = (aov1 - aov0) * orders1        (a different AOV on the new order count)
       volume_effect + aov_effect == revenue1 - revenue0 exactly. This is accounting, not causation."""
    if orders0 in (None, 0) or aov0 is None or orders1 is None or aov1 is None:
        raise AnalyticsError("Cannot split the change: the comparison month has no completed orders.")
    r0, r1 = orders0 * aov0, orders1 * aov1
    volume = (orders1 - orders0) * aov0
    aov_eff = (aov1 - aov0) * orders1
    total = r1 - r0
    return {
        "total_change": total,
        "volume_effect": volume,
        "aov_effect": aov_eff,
        "volume_share_pct": _share(volume, total),
        "aov_share_pct": _share(aov_eff, total),
    }