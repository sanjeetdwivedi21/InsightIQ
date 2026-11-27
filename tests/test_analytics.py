"""

Two kinds of tests:
  1. Toy data small enough to check by hand (formulas are exact).
  2. The real database: results must agree with plain SQL, and the planted scenarios must be found.
Requires the database:  python -m database.setup_database
"""
import json

import numpy as np
import pandas as pd

from database.connection import get_read_only_connection
from src.analytics.business_metrics import kpi_summary, period_metrics
from src.analytics.common import (AnalyticsError, check_month, fmt_money, fmt_pct, pct_change, prev_month,
                                  shift_month, to_py)
from src.analytics.customer_analysis import (customer_behavior, customer_change_buckets, high_value_calibration,
                                             high_value_change, segment_performance, top_customers)
from src.analytics.data_access import load_orders, load_sales
from src.analytics.product_analysis import (category_monthly_revenue, category_performance, decline_streak,
                                            top_products, vs_prior_average_pct)
from src.analytics.revenue_analysis import (decompose_revenue_change, group_change, growth_decline_analysis,
                                            monthly_summary, region_performance)
from src.analytics.revenue_drop import (investigate_revenue_change, is_revenue_change_question,
                                        resolve_month_from_text)


# ====================================================================== toy data helpers
def make_sales(rows):
    """rows: (order_id, month, region, customer, category, revenue, cost)"""
    recs = []
    for i, (oid, month, region, cust, cat, rev, cost) in enumerate(rows, start=1):
        recs.append({"order_item_id": i, "order_id": oid, "order_date": pd.Timestamp(f"{month}-15"),
                     "order_month": month, "order_region": region, "customer_id": cust, "segment": "Consumer",
                     "product_id": 1, "product_name": f"{cat} item", "category": cat, "quantity": 1,
                     "revenue": float(rev), "cost": float(cost), "profit": float(rev - cost),
                     "gross_revenue": float(rev)})
    return pd.DataFrame(recs)


def make_orders(rows):
    """rows: (order_id, customer, month, status)"""
    return pd.DataFrame([{"order_id": o, "customer_id": c, "order_date": pd.Timestamp(f"{m}-15"),
                          "order_month": m, "status": s, "region": "North"} for o, c, m, s in rows])


# Jan: A buys Electronics 1000, B Clothing 500, C Clothing 500        -> 2000, 3 orders
# Feb: A Electronics 400, B Clothing 600, D Furniture 300             -> 1300, 3 orders
TOY = make_sales([
    (1, "2025-01", "North", "A", "Electronics", 1000, 800),
    (2, "2025-01", "South", "B", "Clothing", 500, 250),
    (3, "2025-01", "South", "C", "Clothing", 500, 250),
    (4, "2025-02", "North", "A", "Electronics", 400, 320),
    (5, "2025-02", "South", "B", "Clothing", 600, 300),
    (6, "2025-02", "West", "D", "Furniture", 300, 150),
])
TOY_ORDERS = make_orders([(1, "A", "2025-01", "Completed"), (2, "B", "2025-01", "Completed"),
                          (3, "C", "2025-01", "Completed"), (7, "E", "2025-01", "Cancelled"),
                          (4, "A", "2025-02", "Completed"), (5, "B", "2025-02", "Completed"),
                          (6, "D", "2025-02", "Completed"), (8, "F", "2025-02", "Returned"),
                          (9, "G", "2025-02", "Cancelled"), (10, "H", "2025-02", "Cancelled")])


def _sql_scalar(sql):
    conn = get_read_only_connection()
    try:
        return conn.execute(sql).fetchone()[0]
    finally:
        conn.close()


# ====================================================================== helpers
def test_month_helpers_and_formatting():
    assert prev_month("2025-01") == "2024-12" and shift_month("2025-03", -12) == "2024-03"
    for bad in ["2025-13", "March", "2025-3", None]:
        try:
            check_month(bad)
        except AnalyticsError:
            continue
        raise AssertionError(bad)
    assert pct_change(110, 100) == 10.0 and pct_change(5, 0) is None and pct_change(None, 1) is None
    assert fmt_money(5977000) == "₹59.77 lakh" and fmt_money(170726714) == "₹17.07 crore"
    assert fmt_money(14868) == "₹14,868" and fmt_money(-96394) == "-₹96,394" and fmt_money(None) == "n/a"
    assert fmt_pct(-8.8) == "-8.8%" and fmt_pct(None) == "n/a"


def test_to_py_makes_everything_json_safe():
    data = {"a": np.int64(3), "b": np.float64(2.5), "c": float("nan"), "d": pd.Timestamp("2025-01-02"),
            "e": pd.DataFrame({"x": [1, 2]}), "f": np.bool_(True)}
    assert json.loads(json.dumps(to_py(data))) == {"a": 3, "b": 2.5, "c": None, "d": "2025-01-02",
                                                  "e": [{"x": 1}, {"x": 2}], "f": True}


# ====================================================================== toy data: exact formulas
def test_period_metrics_toy():
    m = period_metrics(TOY, TOY_ORDERS, "2025-01")
    assert m["revenue"] == 2000 and m["orders"] == 3 and abs(m["aov"] - 2000 / 3) < 1e-9
    assert abs(m["margin_pct"] - 100 * 700 / 2000) < 1e-9          # profit 200+250+250 = 700
    assert m["cancelled_orders"] == 1 and m["orders_all"] == 4 and m["cancel_rate_pct"] == 25.0
    f = period_metrics(TOY, TOY_ORDERS, "2025-02")
    assert f["returned_orders"] == 1 and f["cancelled_orders"] == 2 and f["orders_all"] == 6
    assert abs(f["return_rate_pct"] - 100 / 6) < 1e-9 and abs(f["cancel_rate_pct"] - 100 / 3) < 1e-9


def test_group_change_by_category_toy():
    t = group_change(TOY, "2025-02", "2025-01", "category").set_index("category")
    assert t.loc["Electronics", "change"] == -600 and t.loc["Clothing", "change"] == -400
    assert t.loc["Furniture", "change"] == 300 and abs(t.loc["Electronics", "pct_change"] + 60) < 1e-9
    assert abs(t["contribution_pct"].sum() - 100) < 1e-9
    assert abs(t.loc["Electronics", "share_of_decreases_pct"] - 60) < 1e-9
    assert abs(t.loc["Furniture", "share_of_increases_pct"] - 100) < 1e-9
    assert np.isnan(t.loc["Furniture", "pct_change"])               # no earlier revenue: change undefined
    assert list(group_change(TOY, "2025-02", "2025-01", "category")["category"])[0] == "Electronics"


def test_group_change_by_region_toy():
    t = group_change(TOY, "2025-02", "2025-01", "order_region").set_index("order_region")
    assert t.loc["North", "change"] == -600 and t.loc["South", "change"] == -400 and t.loc["West", "change"] == 300


def test_customer_buckets_add_up_toy():
    table, s = customer_change_buckets(TOY, "2025-02", "2025-01")
    status = table.set_index("customer_id")["status"].to_dict()
    assert status == {"A": "reduced", "B": "increased", "C": "lost", "D": "new"}
    assert s["new"]["revenue_effect"] == 300 and s["lost"]["revenue_effect"] == -500
    assert s["reduced"]["revenue_effect"] == -600 and s["increased"]["revenue_effect"] == 100
    assert sum(v["revenue_effect"] for v in s.values()) == -700
    assert abs(sum(v["share_of_net_change_pct"] for v in s.values()) - 100) < 1e-9


def test_revenue_split_is_exact():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n0, n1 = int(rng.integers(1, 500)), int(rng.integers(1, 500))
        a0, a1 = float(rng.uniform(100, 30000)), float(rng.uniform(100, 30000))
        d = decompose_revenue_change(n0, a0, n1, a1)
        assert abs(d["volume_effect"] + d["aov_effect"] - (n1 * a1 - n0 * a0)) < 1e-6
    toy = decompose_revenue_change(3, 2000 / 3, 3, 1300 / 3)
    assert abs(toy["volume_effect"]) < 1e-9 and abs(toy["aov_effect"] + 700) < 1e-9


def test_revenue_split_refuses_empty_comparison_month():
    try:
        decompose_revenue_change(0, None, 5, 100)
    except AnalyticsError:
        return
    raise AssertionError("expected AnalyticsError")


def test_monthly_summary_mom_yoy_and_missing_months():
    rows = []
    oid = 0
    for m, rev in [("2024-01", 100), ("2024-02", 200), ("2024-04", 400), ("2025-01", 150)]:   # 2024-03 missing
        oid += 1
        rows.append((oid, m, "North", "A", "Clothing", rev, rev / 2))
    ms = monthly_summary(make_sales(rows)).set_index("month")
    assert len(ms) == 13 and ms.loc["2024-03", "revenue"] == 0                         # gap filled with zero
    assert abs(ms.loc["2024-02", "mom_change_pct"] - 100) < 1e-9
    assert ms.loc["2024-02", "mom_change_abs"] == 100 and ms.loc["2024-03", "mom_change_abs"] == -200
    assert ms.loc["2024-04", "mom_change_abs"] == 400                                  # 400 vs the zero month
    assert np.isnan(ms.loc["2024-04", "mom_change_pct"])                               # previous month was zero
    assert abs(ms.loc["2025-01", "yoy_change_pct"] - 50) < 1e-9
    assert abs(ms.loc["2024-01", "margin_pct"] - 50) < 1e-9 and ms.loc["2024-01", "aov"] == 100


def test_growth_decline_analysis_toy():
    rows = [(i + 1, f"2025-{i + 1:02d}", "North", "A", "Clothing", r, r / 2)
            for i, r in enumerate([100, 120, 90, 150])]
    g = growth_decline_analysis(monthly_summary(make_sales(rows)))
    assert g["best_month"]["month"] == "2025-04" and g["worst_month"]["month"] == "2025-03"
    assert g["largest_mom_drop"]["month"] == "2025-03" and abs(g["largest_mom_drop"]["pct"] + 25) < 1e-9
    assert g["declining_months"] == 1 and g["rising_months"] == 2 and g["trend_slope_per_month"] > 0
    assert growth_decline_analysis(pd.DataFrame()) == {}


def test_decline_streak_and_prior_average():
    s = pd.Series([10, 12, 11, 9, 8], index=["2025-01", "2025-02", "2025-03", "2025-04", "2025-05"])
    assert decline_streak(s, "2025-05") == 3 and decline_streak(s, "2025-02") == 0
    assert decline_streak(s, "2030-01") == 0
    tie = pd.Series([12, 10, 10, 9], index=["2025-01", "2025-02", "2025-03", "2025-04"])
    assert decline_streak(tie, "2025-04") == 1          # an equal month breaks the streak
    assert decline_streak(tie, "2025-03") == 0
    # prior 3 months of May: Apr 9, Mar 11, Feb 12 -> average 10.667; May is 8 -> -25%
    assert abs(vs_prior_average_pct(s, "2025-05", 3) + 25.0) < 1e-9
    assert vs_prior_average_pct(s, "2025-01", 3) is None             # no earlier months


def test_high_value_change_flags_a_pullback_toy():
    rows, oid = [], 0
    for m in ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]:
        oid += 2
        rows += [(oid, m, "North", "X", "Clothing", 100, 50), (oid + 1, m, "North", "Y", "Clothing", 10, 5)]
    rows += [(99, "2025-07", "North", "X", "Clothing", 20, 10), (100, "2025-07", "North", "Y", "Clothing", 10, 5)]
    t = high_value_change(make_sales(rows), "2025-07", "2025-06", window=6, top_pct=0.5)
    assert list(t["customer_id"]) == ["X"]                                   # top 50% of 2 customers
    r = t.iloc[0]
    assert r["baseline_monthly_avg"] == 100 and r["revenue"] == 20 and r["change_vs_baseline"] == -80
    assert bool(r["sharply_reduced"]) is True and abs(r["pct_vs_baseline"] + 80) < 1e-9
    # a customer at 70% of their usual spend has dipped, but not "sharply" (threshold is 50%)
    mild = [x for x in rows if x[0] not in (99, 100)] + [(99, "2025-07", "North", "X", "Clothing", 70, 35),
                                                         (100, "2025-07", "North", "Y", "Clothing", 10, 5)]
    t2 = high_value_change(make_sales(mild), "2025-07", "2025-06", window=6, top_pct=0.5).iloc[0]
    assert t2["change_vs_baseline"] == -30 and bool(t2["sharply_reduced"]) is False


def test_high_value_calibration_needs_history():
    rows = [(i, "2025-01", "North", "X", "Clothing", 100, 50) for i in range(1, 4)]
    assert high_value_calibration(make_sales(rows), "2025-01") is None


# ====================================================================== toy data: investigation
def test_investigation_toy_report():
    r = investigate_revenue_change(TOY, TOY_ORDERS, "2025-02")
    assert r.direction == "decrease" and r.compare_to == "2025-01"
    assert abs(r.summary["change_pct"] + 35) < 1e-9 and r.summary["change"] == -700
    assert abs(r.decomposition["aov_effect"] + 700) < 1e-9
    assert r.category_table.iloc[0]["category"] == "Electronics"
    ids = [f["id"] for f in r.findings]
    for needed in ["revenue_change", "volume_vs_aov", "category_changes", "customer_buckets", "region_changes",
                   "order_status"]:
        assert needed in ids, needed
    head = r.facts[0]["statement"]
    assert "decreased by 35.0%" in head and "₹2,000" in head and "₹1,300" in head
    assert all(f["type"] in ("fact", "hypothesis") for f in r.findings)
    assert any("observational" in x for x in r.limitations)


def test_investigation_reports_an_increase_honestly():
    r = investigate_revenue_change(TOY, TOY_ORDERS, "2025-01", compare_to="2025-02")
    assert r.direction == "increase" and "increased by" in r.facts[0]["statement"]
    assert "category_reasons" not in [f["id"] for f in r.hypotheses]       # no "why did it drop" guesses
    ids = [f["id"] for f in r.findings]
    assert "category_persistence" not in ids                               # decline-streak talk is for drops
    by_id = {f["id"]: f["statement"] for f in r.findings}
    assert by_id["category_changes"].startswith("Largest category increases: Electronics")
    assert by_id["region_changes"].startswith("Largest regional increase: North")


def test_investigation_input_errors():
    for kwargs in [dict(month="2025-09"), dict(month="2025-02", compare_to="2025-02"), dict(month="Feb")]:
        try:
            investigate_revenue_change(TOY, TOY_ORDERS, **kwargs)
        except AnalyticsError as exc:
            assert str(exc)
            continue
        raise AssertionError(kwargs)
    try:
        investigate_revenue_change(TOY.drop(columns=["category"]), TOY_ORDERS, "2025-02")
    except AnalyticsError as exc:
        assert "category" in str(exc)
    else:
        raise AssertionError("missing column not detected")


def test_report_is_json_serializable():
    d = investigate_revenue_change(TOY, TOY_ORDERS, "2025-02").to_dict()
    assert json.loads(json.dumps(d))["summary"]["change"] == -700


def test_question_detection_and_month_resolution():
    for q in ["Why did revenue decrease in March?", "why did sales drop last month", "Explain the fall in revenue"]:
        assert is_revenue_change_question(q), q
    for q in ["What is our total revenue?", "Which category has the highest sales?", "Show monthly revenue", ""]:
        assert not is_revenue_change_question(q), q
    months = ["2024-02", "2024-03", "2025-02", "2025-03", "2025-05"]
    assert resolve_month_from_text("Why did revenue decrease in March?", months) == "2025-03"   # latest March
    assert resolve_month_from_text("why did sales drop in March 2024", months) == "2024-03"
    assert resolve_month_from_text("explain 2025-02", months) == "2025-02"
    assert resolve_month_from_text("sales in June", months) is None
    assert resolve_month_from_text("revenue may drop soon", months) is None            # 'may' the verb
    assert resolve_month_from_text("why was revenue low in May?", months) == "2025-05"
    assert resolve_month_from_text("No month here", months) is None


# ====================================================================== real database
def test_totals_agree_with_plain_sql():
    sales, orders = load_sales(), load_orders()
    assert abs(monthly_summary(sales)["revenue"].sum() - _sql_scalar("SELECT SUM(revenue) FROM vw_sales")) < 1.0
    k = kpi_summary(sales, orders)
    assert abs(k["profit"] - _sql_scalar("SELECT SUM(profit) FROM vw_sales")) < 1.0
    assert k["orders"] == _sql_scalar("SELECT COUNT(DISTINCT order_id) FROM vw_sales")
    assert k["orders_all"] == _sql_scalar("SELECT COUNT(*) FROM orders")
    assert abs(k["aov"] - k["revenue"] / k["orders"]) < 1e-6
    feb = period_metrics(sales, orders, "2025-02")["revenue"]
    assert abs(feb - _sql_scalar("SELECT SUM(revenue) FROM vw_sales WHERE order_month = '2025-02'")) < 1.0


def test_group_tables_reconcile_with_sql():
    sales = load_sales()
    cats = category_performance(sales)
    assert abs(cats["revenue"].sum() - sales["revenue"].sum()) < 1.0 and abs(cats["revenue_share_pct"].sum() - 100) < 1e-6
    assert cats.iloc[0]["category"] == "Electronics"
    reg = region_performance(sales)
    assert reg.iloc[-1]["order_region"] == "Central" and abs(reg["share_pct"].sum() - 100) < 1e-6
    assert abs(segment_performance(sales)["revenue"].sum() - sales["revenue"].sum()) < 1.0
    top = top_customers(sales, 10)
    sql_top = _sql_scalar("SELECT SUM(r) FROM (SELECT SUM(revenue) r FROM vw_sales GROUP BY customer_id "
                          "ORDER BY r DESC LIMIT 10)")
    assert abs(top["revenue"].sum() - sql_top) < 1.0
    assert top_products(sales, 1, by="profit").iloc[0]["product_name"] == _sql_scalar(
        "SELECT product_name FROM vw_sales GROUP BY product_id ORDER BY SUM(profit) DESC LIMIT 1")
    assert category_monthly_revenue(sales).shape == (24, 6)


def test_customer_behavior_summary():
    sales = load_sales()
    b = customer_behavior(sales)
    assert b["customers"] == sales["customer_id"].nunique()
    assert 0 < b["repeat_customer_rate_pct"] <= 100 and b["avg_days_between_orders"] > 0
    assert b["top_10pct_customers_revenue_share_pct"] > 10 and b["customers_for_80pct_of_revenue"] <= b["customers"]


def test_trend_summary_matches_known_pattern():
    g = growth_decline_analysis(monthly_summary(load_sales()))
    assert g["months"] == 24 and g["best_month"]["month"] == "2024-11"
    assert g["yearly_change_pct"]["2024->2025"] < 0                    # 2025 revenue below 2024 (Electronics decline)


def test_march_2025_investigation_finds_the_planted_story():
    sales, orders = load_sales(), load_orders()
    r = investigate_revenue_change(sales, orders, "2025-03")
    s = r.summary
    assert r.direction == "decrease" and -9.0 < s["change_pct"] < -8.6
    # the exact accounting identities hold on real data
    assert abs(r.decomposition["volume_effect"] + r.decomposition["aov_effect"] - s["change"]) < 1e-6
    assert abs(r.category_table["change"].sum() - s["change"]) < 1e-6
    assert abs(r.region_table["change"].sum() - s["change"]) < 1e-6
    assert abs(sum(v["revenue_effect"] for v in r.customer_buckets.values()) - s["change"]) < 1e-6
    # Electronics is the main decliner and has been falling for several months
    top = r.category_table.iloc[0]
    assert top["category"] == "Electronics" and top["share_of_decreases_pct"] > 80
    assert r.context["decline_persistence"]["Electronics"]["decline_streak_months"] >= 3
    # several of the 8 planted high-value "reducers" (customer ids 1-8) are among the biggest drops
    biggest = r.high_value_table.sort_values("change_vs_baseline").head(10)["customer_id"].tolist()
    assert sum(1 for c in biggest if c <= 8) >= 3
    # honest framing: overall change is within normal variation, hypotheses are flagged as such
    assert abs(r.context["change_in_standard_deviations"]) < 1
    assert r.hypotheses and all(h["type"] == "hypothesis" for h in r.hypotheses)
    assert any("within normal month-to-month variation" in x for x in r.limitations)
    assert all("because" not in f["statement"].lower() for f in r.findings)      # no causal claims
    json.dumps(r.to_dict())


def test_question_to_report_flow_on_real_months():
    sales, orders = load_sales(), load_orders()
    months = sorted(sales["order_month"].unique())
    q = "Why did revenue decrease in March?"
    assert is_revenue_change_question(q)
    month = resolve_month_from_text(q, months)
    assert month == "2025-03" and investigate_revenue_change(sales, orders, month).direction == "decrease"
    inc = investigate_revenue_change(sales, orders, "2025-05")
    assert inc.direction == "increase" and "increased by" in inc.facts[0]["statement"]
    texts = {f["id"]: f["statement"] for f in inc.findings}
    assert texts["category_changes"].startswith("Largest category increases")
    assert texts["region_changes"].startswith("Largest regional increase")
    assert "category_persistence" not in texts
    dec = investigate_revenue_change(sales, orders, "2025-03")                  # a drop still reads as before
    dtexts = {f["id"]: f["statement"] for f in dec.findings}
    assert dtexts["category_changes"].startswith("Largest category decreases: Electronics")
    assert "category_persistence" in dtexts


def test_analytics_never_changes_the_database():
    before = _sql_scalar("SELECT COUNT(*) FROM orders") + _sql_scalar("SELECT COUNT(*) FROM order_items")
    investigate_revenue_change(load_sales(), load_orders(), "2025-03")
    after = _sql_scalar("SELECT COUNT(*) FROM orders") + _sql_scalar("SELECT COUNT(*) FROM order_items")
    assert before == after