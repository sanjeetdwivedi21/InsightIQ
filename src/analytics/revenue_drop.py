"""

Automated revenue-change investigation ("Why did revenue decrease in March?").

Everything is computed from the data. The output separates:
  * FACTS        - numbers and arithmetic that follow directly from the database
  * HYPOTHESES   - possible reasons worth investigating; clearly marked as NOT confirmed
  * LIMITATIONS  - what this analysis cannot tell you

The code never claims that something CAUSED the change: it quantifies how much each category,
region and customer group contributed to it (an exact accounting split) and reports patterns
(for example, repeated monthly declines) that make a one-off explanation less likely.

Try it:   python -m src.analytics.revenue_drop --month 2025-03
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.analytics.business_metrics import period_metrics
from src.analytics.common import (AnalyticsError, check_month, fmt_money, fmt_pct, pct_change, prev_month,
                                  require_columns, shift_month, to_py)
from src.analytics.customer_analysis import customer_change_buckets, high_value_calibration, high_value_change
from src.analytics.product_analysis import (category_change, category_monthly_revenue, decline_streak,
                                            vs_prior_average_pct)
from src.analytics.revenue_analysis import (decompose_revenue_change, group_change, monthly_summary,
                                            region_change)

_SALES_COLS = ["order_id", "order_month", "order_region", "customer_id", "category", "quantity", "revenue",
               "profit", "gross_revenue"]
_ORDER_COLS = ["order_id", "order_month", "status"]


@dataclass
class RevenueChangeReport:
    month: str
    compare_to: str
    direction: str                      # decrease | increase | flat
    summary: dict
    decomposition: dict | None
    aov_drivers: dict
    category_table: pd.DataFrame
    region_table: pd.DataFrame
    segment_table: pd.DataFrame
    customer_buckets: dict
    customer_table: pd.DataFrame        # biggest customer-level drops (top_n)
    high_value: dict
    high_value_table: pd.DataFrame
    context: dict
    findings: list = field(default_factory=list)       # each: {id, type: fact|hypothesis, statement, metrics}
    limitations: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-friendly dictionary (DataFrames become lists of records)."""
        return to_py({k: getattr(self, k) for k in self.__dataclass_fields__})

    @property
    def facts(self) -> list:
        return [f for f in self.findings if f["type"] == "fact"]

    @property
    def hypotheses(self) -> list:
        return [f for f in self.findings if f["type"] == "hypothesis"]


# --------------------------------------------------------------------------- question helpers
_MONTH_NAMES = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
                "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_MONTH_RE = re.compile(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
                       r"sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b(?:\s*,?\s*(20\d{2}))?",
                       re.IGNORECASE)
_ISO_RE = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])\b")
_WHY_RE = re.compile(r"\b(why|reasons?|causes?|explain|what happened|how come)\b", re.IGNORECASE)
_METRIC_RE = re.compile(r"\b(revenue|sales|income|turnover)\b", re.IGNORECASE)
_MOVE_RE = re.compile(r"(decreas|drop|declin|fall|fell|down|dip|slump|lower|reduc|increas|rise|rose|grow|jump|spike|surge)",
                      re.IGNORECASE)


def is_revenue_change_question(text: str) -> bool:
    """True for questions like 'Why did revenue decrease in March?' (asks why + revenue + a movement)."""
    return bool(text and _WHY_RE.search(text) and _METRIC_RE.search(text) and _MOVE_RE.search(text))


def resolve_month_from_text(text: str, available_months: list[str]) -> str | None:
    """Find the month a question is about. 'March 2025' or '2025-03' are exact; plain 'March' means the
    latest March in the data. Returns None if no month is mentioned or the month has no data."""
    if not text:
        return None
    avail = set(available_months)
    iso = _ISO_RE.search(text)
    if iso:
        m = f"{iso.group(1)}-{iso.group(2)}"
        return m if m in avail else None
    for match in _MONTH_RE.finditer(text):
        word, year = match.group(1).lower(), match.group(2)
        num = _MONTH_NAMES[word[:3]]
        if word == "may" and not year and not re.search(r"\b(?:in|for|during|of|from|since|than|to)\s+may\b", text, re.I):
            continue                                   # "may" the verb, not the month
        suffix = f"-{num:02d}"
        if year:
            m = f"{year}{suffix}"
            return m if m in avail else None
        candidates = sorted(m for m in avail if m.endswith(suffix))
        return candidates[-1] if candidates else None
    return None


# --------------------------------------------------------------------------- main analysis
def _direction(change: float) -> str:
    return "decrease" if change < 0 else ("increase" if change > 0 else "flat")


def _records(df: pd.DataFrame, n: int) -> list[dict]:
    return to_py(df.head(n))


def investigate_revenue_change(sales: pd.DataFrame, orders: pd.DataFrame, month: str,
                               compare_to: str | None = None, top_n: int = 5, hv_window: int = 6,
                               hv_top_pct: float = 0.05) -> RevenueChangeReport:
    """Explain how revenue in `month` differs from `compare_to` (default: the month before)."""
    require_columns(sales, _SALES_COLS, "sales")
    require_columns(orders, _ORDER_COLS, "orders")
    month = check_month(month)
    prev = check_month(compare_to) if compare_to else prev_month(month)
    if prev == month:
        raise AnalyticsError("Choose two different months to compare.")
    available = sorted(sales["order_month"].unique())
    if not available:
        raise AnalyticsError("There is no sales data to analyse.")
    for m in (month, prev):
        if m not in available:
            raise AnalyticsError(f"No sales data for {m}. The data covers {available[0]} to {available[-1]}.")

    cur, base = period_metrics(sales, orders, month), period_metrics(sales, orders, prev)
    change = cur["revenue"] - base["revenue"]
    change_pct = pct_change(cur["revenue"], base["revenue"])
    direction = _direction(change)
    summary = {
        "month": month, "compare_to": prev, "direction": direction,
        "revenue": cur["revenue"], "prev_revenue": base["revenue"], "change": change, "change_pct": change_pct,
        "orders": cur["orders"], "prev_orders": base["orders"],
        "orders_change_pct": pct_change(cur["orders"], base["orders"]),
        "aov": cur["aov"], "prev_aov": base["aov"], "aov_change_pct": pct_change(cur["aov"], base["aov"]),
        "profit": cur["profit"], "prev_profit": base["profit"],
        "profit_change_pct": pct_change(cur["profit"], base["profit"]),
        "margin_pct": cur["margin_pct"], "prev_margin_pct": base["margin_pct"],
    }

    decomposition = None
    if base["orders"] and cur["orders"] and base["aov"] is not None and cur["aov"] is not None:
        decomposition = decompose_revenue_change(base["orders"], base["aov"], cur["orders"], cur["aov"])

    aov_drivers = {
        "items_per_order": {"prev": base["items_per_order"], "cur": cur["items_per_order"],
                            "pct_change": pct_change(cur["items_per_order"], base["items_per_order"])},
        "revenue_per_item": {"prev": base["revenue_per_item"], "cur": cur["revenue_per_item"],
                             "pct_change": pct_change(cur["revenue_per_item"], base["revenue_per_item"])},
        "discount_pct": {"prev": base["discount_pct"], "cur": cur["discount_pct"],
                         "point_change": None if base["discount_pct"] is None or cur["discount_pct"] is None
                         else cur["discount_pct"] - base["discount_pct"]},
    }

    cat_table = category_change(sales, month, prev)
    region_table = region_change(sales, month, prev)
    segment_table = group_change(sales, month, prev, "segment") if "segment" in sales.columns else pd.DataFrame()
    buckets_table, buckets = customer_change_buckets(sales, month, prev)
    hv_table = high_value_change(sales, month, prev, window=hv_window, top_pct=hv_top_pct)
    hv = {
        "window_months": hv_window, "top_pct": hv_top_pct,
        "customers": int(len(hv_table)),
        "sharply_reduced": int(hv_table["sharply_reduced"].sum()) if len(hv_table) else 0,
        "baseline_monthly_total": float(hv_table["baseline_monthly_avg"].sum()) if len(hv_table) else 0.0,
        "current_total": float(hv_table["revenue"].sum()) if len(hv_table) else 0.0,
    }
    hv["change_vs_baseline"] = hv["current_total"] - hv["baseline_monthly_total"]
    hv["share_of_net_change_pct"] = None if change == 0 else hv["change_vs_baseline"] / change * 100
    hv["calibration"] = high_value_calibration(sales, month, window=hv_window, top_pct=hv_top_pct)

    # category persistence evidence for the biggest decliners
    cat_monthly = category_monthly_revenue(sales)
    persistence = {}
    for cat in cat_table.loc[cat_table["change"] < 0, "category"].head(top_n):
        persistence[cat] = {"decline_streak_months": decline_streak(cat_monthly[cat], month),
                            "pct_vs_prior_3m_avg": vs_prior_average_pct(cat_monthly[cat], month, 3)}

    # context: seasonality, normal volatility
    ms = monthly_summary(sales).set_index("month")
    py_prev, py_cur = shift_month(prev, -12), shift_month(month, -12)
    prior_year_pct = None
    if py_prev in ms.index and py_cur in ms.index:
        prior_year_pct = pct_change(ms.loc[py_cur, "revenue"], ms.loc[py_prev, "revenue"])
    history = ms.loc[ms.index < month, "mom_change_pct"].dropna()
    typical_swing = float(history.std(ddof=1)) if len(history) >= 6 else None
    z = None if (typical_swing in (None, 0) or change_pct is None) else change_pct / typical_swing
    context = {
        "prior_year_same_transition_pct": prior_year_pct,
        "prior_year_months": [py_prev, py_cur],
        "typical_monthly_swing_pct": typical_swing,
        "change_in_standard_deviations": z,
        "cancel_rate_pct": {"prev": base["cancel_rate_pct"], "cur": cur["cancel_rate_pct"]},
        "return_rate_pct": {"prev": base["return_rate_pct"], "cur": cur["return_rate_pct"]},
        "decline_persistence": persistence,
    }

    report = RevenueChangeReport(
        month=month, compare_to=prev, direction=direction, summary=summary, decomposition=decomposition,
        aov_drivers=aov_drivers, category_table=cat_table, region_table=region_table,
        segment_table=segment_table, customer_buckets=buckets,
        customer_table=buckets_table.head(top_n), high_value=hv, high_value_table=hv_table.head(top_n),
        context=context)
    report.findings = _build_findings(report, cur, base, top_n)
    report.limitations = _build_limitations(report, cur, base)
    return report


# --------------------------------------------------------------------------- findings
def _fact(fid: str, text: str, **metrics) -> dict:
    return {"id": fid, "type": "fact", "statement": text, "metrics": to_py(metrics)}


def _hyp(fid: str, text: str, **metrics) -> dict:
    return {"id": fid, "type": "hypothesis", "statement": text, "metrics": to_py(metrics)}


def _build_findings(r: RevenueChangeReport, cur: dict, base: dict, top_n: int) -> list:
    s, prev, month = r.summary, r.compare_to, r.month
    out: list[dict] = []

    verb = {"decrease": "decreased", "increase": "increased", "flat": "was unchanged"}[r.direction]
    if s["change_pct"] is None:
        head = f"Revenue went from {fmt_money(s['prev_revenue'])} in {prev} to {fmt_money(s['revenue'])} in {month}."
    elif r.direction == "flat":
        head = f"Revenue was unchanged between {prev} and {month} ({fmt_money(s['revenue'])})."
    else:
        head = (f"Revenue {verb} by {abs(s['change_pct']):.1f}% from {prev} to {month} "
                f"({fmt_money(s['prev_revenue'])} to {fmt_money(s['revenue'])}; net change {fmt_money(s['change'])}).")
    out.append(_fact("revenue_change", head, **{k: s[k] for k in ("prev_revenue", "revenue", "change", "change_pct")}))

    if r.decomposition:
        d = r.decomposition
        shares = ""
        if d["volume_share_pct"] is not None and d["aov_share_pct"] is not None:
            shares = (f" ({d['volume_share_pct']:.0f}% and {d['aov_share_pct']:.0f}% of the net change)")
        out.append(_fact(
            "volume_vs_aov",
            f"Completed orders went from {s['prev_orders']} to {s['orders']} ({fmt_pct(s['orders_change_pct'])}) and "
            f"average order value from {fmt_money(s['prev_aov'])} to {fmt_money(s['aov'])} ({fmt_pct(s['aov_change_pct'])}). "
            f"Arithmetic split of the net change: order volume {fmt_money(d['volume_effect'])}, average order value "
            f"{fmt_money(d['aov_effect'])}{shares}. This is an accounting split, not proof of cause.", **d))

    ct = r.category_table
    if len(ct):
        dec = ct[ct["change"] < 0]
        inc = ct[ct["change"] > 0].sort_values("change", ascending=False)
        if r.direction == "increase":
            parts = [f"{row['category']} {fmt_pct(row['pct_change'])} ({fmt_money(row['change'])}, "
                     f"{row['share_of_increases_pct']:.0f}% of all category increases)" for _, row in inc.head(2).iterrows()]
            sentence = ("Largest category increases: " + "; ".join(parts) + ".") if parts else "No category increased."
        else:
            parts = [f"{row['category']} {fmt_pct(row['pct_change'])} ({fmt_money(row['change'])}, "
                     f"{row['share_of_decreases_pct']:.0f}% of all category decreases)" for _, row in dec.head(2).iterrows()]
            sentence = ("Largest category decreases: " + "; ".join(parts) + ".") if parts else "No category decreased."
        sentence += (f" Category decreases total {fmt_money(dec['change'].sum())} and increases total "
                     f"{fmt_money(inc['change'].sum())}; they net to {fmt_money(ct['change'].sum())}.")
        out.append(_fact("category_changes", sentence, decreases=_records(dec, top_n), increases=_records(inc, top_n)))
        if len(dec) and r.direction == "decrease":      # persistence matters when explaining a drop
            top = dec.iloc[0]["category"]
            p = r.context["decline_persistence"].get(top, {})
            if p.get("decline_streak_months", 0) >= 1 and p.get("pct_vs_prior_3m_avg") is not None:
                out.append(_fact(
                    "category_persistence",
                    f"{top} revenue has fallen month over month for {p['decline_streak_months']} consecutive "
                    f"month(s) up to {month}, and is {fmt_pct(p['pct_vs_prior_3m_avg'])} versus its average of the "
                    f"previous 3 months.", category=top, **p))

    b = r.customer_buckets
    out.append(_fact(
        "customer_buckets",
        f"Of customers who bought in either month: {b['lost']['customers']} bought in {prev} but not in {month} "
        f"({fmt_money(b['lost']['revenue_effect'])}); {b['new']['customers']} bought in {month} but not in {prev} "
        f"({fmt_money(b['new']['revenue_effect'])}); among customers who bought in both, {b['reduced']['customers']} "
        f"spent less ({fmt_money(b['reduced']['revenue_effect'])}) and {b['increased']['customers']} spent more "
        f"({fmt_money(b['increased']['revenue_effect'])}). Individual customers buying in some months and not others "
        f"is normal, so only large, concentrated drops are meaningful.", **b))

    hv = r.high_value
    if hv["customers"]:
        cal = hv["calibration"]
        worst = r.high_value_table[r.high_value_table["change_vs_baseline"] < 0].head(3)
        worst_txt = ", ".join(f"customer {x.customer_id} ({fmt_money(x.change_vs_baseline)} vs their usual "
                              f"monthly spend of {fmt_money(x.baseline_monthly_avg)})" for x in worst.itertuples())
        text = (f"Among the top {hv['top_pct'] * 100:.0f}% of customers by revenue over the {hv['window_months']} "
                f"months before {month} ({hv['customers']} customers), combined spend in {month} was "
                f"{fmt_money(hv['current_total'])} versus a usual {fmt_money(hv['baseline_monthly_total'])} per month "
                f"({fmt_pct(None if not hv['baseline_monthly_total'] else hv['change_vs_baseline'] / hv['baseline_monthly_total'] * 100)}); "
                f"{hv['sharply_reduced']} of them spent less than half their usual amount.")
        if cal:
            z = cal["cohort_z"]
            z_txt = "" if z is None else f" This is {z:+.1f} standard deviations from typical."
            text += (f" High-value customers buy in lumps, so they normally fall short of their own average in a "
                     f"given month: in {cal['history_months']} earlier months the typical result was "
                     f"{fmt_pct(cal['typical_cohort_pct'])} and {cal['typical_sharp_share_pct']:.0f}% of the group "
                     f"spent under half their usual amount (this month: {cal['sharp_share_pct']:.0f}%).{z_txt} "
                     f"The group has been below its typical level for {cal['months_below_typical_in_a_row']} "
                     f"consecutive month(s).")
        else:
            text += " There is not enough earlier history to tell how unusual this is."
        if worst_txt:
            text += f" Largest individual drops: {worst_txt}."
        out.append(_fact("high_value_customers", text, **hv, largest_drops=_records(r.high_value_table, top_n)))

    rt = r.region_table
    if len(rt):
        if r.direction == "increase":
            reg_inc = rt[rt["change"] > 0].sort_values("change", ascending=False)
            if len(reg_inc):
                w = reg_inc.iloc[0]
                out.append(_fact("region_changes",
                                 f"Largest regional increase: {w['order_region']} {fmt_pct(w['pct_change'])} "
                                 f"({fmt_money(w['change'])}, {w['share_of_increases_pct']:.0f}% of all regional increases).",
                                 regions=_records(rt, 10)))
            else:
                out.append(_fact("region_changes", "No region had higher revenue than the comparison month.",
                                 regions=_records(rt, 10)))
        else:
            reg_dec = rt[rt["change"] < 0]
            if len(reg_dec):
                w = reg_dec.iloc[0]
                out.append(_fact("region_changes",
                                 f"Largest regional decrease: {w['order_region']} {fmt_pct(w['pct_change'])} "
                                 f"({fmt_money(w['change'])}, {w['share_of_decreases_pct']:.0f}% of all regional decreases).",
                                 regions=_records(rt, 10)))
            else:
                out.append(_fact("region_changes", "No region had lower revenue than the comparison month.",
                                 regions=_records(rt, 10)))

    c = r.context
    cr, rr = c["cancel_rate_pct"], c["return_rate_pct"]
    if None not in (cr["prev"], cr["cur"], rr["prev"], rr["cur"]):
        out.append(_fact("order_status",
                         f"Cancelled orders were {cr['prev']:.1f}% of all orders in {prev} and {cr['cur']:.1f}% in "
                         f"{month}; returned orders were {rr['prev']:.1f}% and {rr['cur']:.1f}%.", **c))

    py = c["prior_year_same_transition_pct"]
    if py is not None and s["change_pct"] is not None:
        same = (py < 0) == (s["change_pct"] < 0)
        out.append(_fact("seasonality_reference",
                         f"A year earlier, the same transition ({c['prior_year_months'][0]} to "
                         f"{c['prior_year_months'][1]}) changed revenue by {fmt_pct(py)}, "
                         f"{'in the same direction as' if same else 'in the opposite direction to'} this year's "
                         f"{fmt_pct(s['change_pct'])}.", prior_year_pct=py, this_year_pct=s["change_pct"]))

    z = c["change_in_standard_deviations"]
    if z is not None:
        size = "within" if abs(z) < 1 else ("somewhat larger than" if abs(z) < 2 else "much larger than")
        out.append(_fact("normal_variation",
                         f"Month-over-month revenue in this data typically moves by about ±"
                         f"{c['typical_monthly_swing_pct']:.1f}% (standard deviation). This change is {z:+.1f} "
                         f"standard deviations, which is {size} the usual range of monthly fluctuation.",
                         z=z, typical_swing_pct=c["typical_monthly_swing_pct"]))

    # ---- hypotheses (never presented as established) ----
    if len(ct) and r.direction == "decrease":
        top = ct.iloc[0]
        if top["change"] < 0 and top["share_of_decreases_pct"] >= 40:
            out.append(_hyp("category_reasons",
                            f"Possible reasons to investigate for {top['category']} (hypotheses, not confirmed): "
                            f"pricing changes, stock availability, a shift in demand, competitor activity, or the "
                            f"customer changes above. This database has no pricing-history, stock, marketing or "
                            f"competitor data, so none of these can be confirmed here.", category=top["category"]))
    if r.decomposition and r.decomposition["aov_share_pct"] is not None and abs(r.decomposition["aov_share_pct"]) >= 50:
        d = r.aov_drivers
        movers = {k: d[k]["pct_change"] for k in ("items_per_order", "revenue_per_item") if d[k]["pct_change"] is not None}
        if movers:
            lead = max(movers, key=lambda k: abs(movers[k]))
            label = {"items_per_order": "items per order", "revenue_per_item": "revenue per item sold"}[lead]
            disc = d["discount_pct"]["point_change"]
            disc_txt = "" if disc is None or abs(disc) < 1 else f"; average discount moved {disc:+.1f} percentage points"
            out.append(_hyp("aov_reasons",
                            f"The average-order-value movement coincides most with {label} ({fmt_pct(movers[lead])})"
                            f"{disc_txt}. Whether this caused the change cannot be established from this data.",
                            **{k: d[k] for k in d}))
    cal = hv.get("calibration")
    if (r.direction == "decrease" and cal and cal["cohort_z"] is not None and cal["cohort_z"] <= -1
            and hv["change_vs_baseline"] < 0):
        out.append(_hyp("high_value_reasons",
                        "Spending by high-value customers is lower than is typical for this group, which may reflect "
                        "churn, project or contract timing, or switching to other suppliers; the database holds no "
                        "reason data for individual customers.", cohort_z=cal["cohort_z"]))
    return out


def _build_limitations(r: RevenueChangeReport, cur: dict, base: dict) -> list[str]:
    lim = [
        "Revenue means Completed orders after discounts; cancelled and returned orders are excluded.",
        "This is an observational comparison of two months. Patterns in the data show association, not proven cause.",
        "The database has no data on pricing history, promotions, stock levels, marketing or competitors.",
    ]
    z = r.context["change_in_standard_deviations"]
    if z is None:
        lim.append("There is not enough earlier history to judge what a normal month-to-month change looks like.")
    elif abs(z) < 1:
        lim.append("The change is within normal month-to-month variation, so part of it may be random fluctuation.")
    if r.high_value["customers"] and not r.high_value.get("calibration"):
        lim.append("There is not enough earlier history to calibrate the high-value customer comparison.")
    if min(cur["orders"], base["orders"]) < 30:
        lim.append("One of the months has fewer than 30 completed orders, which makes comparisons noisy.")
    return lim


# --------------------------------------------------------------------------- plain-text output
def format_report(r: RevenueChangeReport) -> str:
    lines = [f"REVENUE CHANGE INVESTIGATION: {r.compare_to} -> {r.month}", "=" * 60, "", "FACTS (from the data)"]
    lines += [f"- {f['statement']}" for f in r.facts]
    if r.hypotheses:
        lines += ["", "HYPOTHESES (NOT confirmed)"]
        lines += [f"- {f['statement']}" for f in r.hypotheses]
    lines += ["", "LIMITATIONS"] + [f"- {x}" for x in r.limitations]
    return "\n".join(lines)


def main() -> int:
    from src.analytics.data_access import load_orders, load_sales

    parser = argparse.ArgumentParser(description="Explain a month's revenue change")
    parser.add_argument("--month", required=True, help="month to explain, e.g. 2025-03")
    parser.add_argument("--compare-to", help="month to compare with (default: the month before)")
    args = parser.parse_args()
    try:
        report = investigate_revenue_change(load_sales(), load_orders(), args.month, args.compare_to)
    except AnalyticsError as exc:
        print(f"Error: {exc}")
        return 1
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())