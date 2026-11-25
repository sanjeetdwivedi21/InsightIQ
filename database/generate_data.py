"""
database/generate_data.py

Generates a realistic synthetic retail dataset (amounts in INR) and writes CSVs
to data/raw/. Run from the project root:

    python -m database.generate_data

Planted business scenarios (so questions like "Why did revenue drop in March?"
have a real answer in the data):
  1. Electronics sales decline steadily through 2025.
  2. Eight high-value (VIP) customers sharply reduce purchases from Feb-Mar 2025.
  3. Seasonal peaks in Oct/Nov (festive season) and higher discounts then.
  4. Regions differ: Central is smallest, North largest.
  5. Average order value varies by month (discounts, basket size, category mix).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

START_DATE = date(2024, 1, 1)
END_DATE = date(2025, 12, 31)

N_CUSTOMERS = 2200
N_VIP = 30            # high-value customers
N_VIP_REDUCERS = 8    # VIPs who cut back in 2025
BASE_ORDERS_PER_MONTH = 440

REGIONS = {
    "North":   (0.28, ["Delhi", "Lucknow", "Jaipur", "Chandigarh"]),
    "South":   (0.24, ["Bengaluru", "Chennai", "Hyderabad", "Kochi"]),
    "West":    (0.22, ["Mumbai", "Pune", "Ahmedabad", "Surat"]),
    "East":    (0.16, ["Kolkata", "Patna", "Bhubaneswar", "Guwahati"]),
    "Central": (0.10, ["Bhopal", "Indore", "Nagpur", "Raipur"]),
}
SEGMENTS = {"Consumer": 0.60, "Corporate": 0.25, "Small Business": 0.15}

# category: (share of items, price range INR, cost-to-price ratio range)
CATEGORIES = {
    "Electronics":    (0.22, (1500, 80000), (0.78, 0.88)),
    "Furniture":      (0.10, (2000, 40000), (0.60, 0.75)),
    "Clothing":       (0.28, (500, 6000),   (0.45, 0.60)),
    "Home & Kitchen": (0.18, (400, 12000),  (0.55, 0.70)),
    "Sports":         (0.12, (600, 15000),  (0.55, 0.70)),
    "Beauty":         (0.10, (200, 4000),   (0.35, 0.50)),
}
SUB_CATEGORIES = {
    "Electronics":    ["Mobiles", "Laptops", "Audio", "Accessories", "Wearables"],
    "Furniture":      ["Sofas", "Beds", "Tables", "Chairs", "Storage"],
    "Clothing":       ["Men", "Women", "Kids", "Footwear", "Winter Wear"],
    "Home & Kitchen": ["Cookware", "Appliances", "Decor", "Bedding", "Cleaning"],
    "Sports":         ["Fitness", "Cricket", "Cycling", "Yoga", "Outdoor"],
    "Beauty":         ["Skincare", "Haircare", "Makeup", "Fragrance", "Grooming"],
}
N_PRODUCTS = 120

# Month-of-year demand multipliers (festive peak in Oct/Nov).
SEASONALITY = {1: 0.90, 2: 0.95, 3: 0.95, 4: 0.92, 5: 0.92, 6: 0.88,
               7: 0.90, 8: 1.00, 9: 1.05, 10: 1.25, 11: 1.35, 12: 1.20}

# Electronics demand multiplier by month of 2025 (the declining category).
ELECTRONICS_DECLINE_2025 = {1: 0.95, 2: 0.88, 3: 0.62, 4: 0.58, 5: 0.55, 6: 0.52,
                            7: 0.50, 8: 0.48, 9: 0.46, 10: 0.50, 11: 0.52, 12: 0.45}
# Activity multiplier for the VIP "reducers" by month of 2025.
VIP_REDUCER_2025 = {1: 1.0, 2: 0.55, 3: 0.12, 4: 0.12, 5: 0.15, 6: 0.15,
                    7: 0.15, 8: 0.15, 9: 0.18, 10: 0.2, 11: 0.2, 12: 0.2}

PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash on Delivery", "Wallet"]
PAYMENT_PROBS = [0.38, 0.22, 0.15, 0.08, 0.12, 0.05]

FIRST_NAMES = ["Aarav", "Vivaan", "Aditya", "Ishaan", "Rohan", "Karan", "Arjun", "Rahul", "Amit", "Sanjay",
               "Priya", "Ananya", "Diya", "Neha", "Pooja", "Kavya", "Riya", "Sneha", "Meera", "Anjali"]
LAST_NAMES = ["Sharma", "Verma", "Gupta", "Singh", "Patel", "Reddy", "Nair", "Iyer", "Das", "Khan",
              "Mishra", "Dwivedi", "Joshi", "Mehta", "Bose", "Kulkarni", "Yadav", "Pandey", "Chopra", "Rao"]

logger = logging.getLogger(__name__)


def _random_dates(rng: np.random.Generator, start: date, end: date, n: int) -> list[date]:
    span = (end - start).days
    return [start + timedelta(days=int(d)) for d in rng.integers(0, span + 1, n)]


def generate_customers(rng: np.random.Generator) -> pd.DataFrame:
    region_names = list(REGIONS)
    region_probs = [REGIONS[r][0] for r in region_names]
    regions = rng.choice(region_names, N_CUSTOMERS, p=region_probs)
    cities = [rng.choice(REGIONS[r][1]) for r in regions]
    segments = rng.choice(list(SEGMENTS), N_CUSTOMERS, p=list(SEGMENTS.values()))

    # The first N_VIP customers are Corporate high-value accounts.
    segments[:N_VIP] = "Corporate"

    # ~60% signed up before the data window starts, rest during it.
    before = _random_dates(rng, date(2022, 1, 1), date(2023, 12, 31), N_CUSTOMERS)
    during = _random_dates(rng, START_DATE, date(2025, 10, 31), N_CUSTOMERS)
    use_before = rng.random(N_CUSTOMERS) < 0.60
    signup = [b if u else d for b, d, u in zip(before, during, use_before)]
    # VIPs are long-standing accounts.
    for i in range(N_VIP):
        signup[i] = date(2022, 1, 1) + timedelta(days=int(rng.integers(0, 600)))

    first = rng.choice(FIRST_NAMES, N_CUSTOMERS)
    last = rng.choice(LAST_NAMES, N_CUSTOMERS)
    ids = np.arange(1, N_CUSTOMERS + 1)
    df = pd.DataFrame({
        "customer_id": ids,
        "first_name": first,
        "last_name": last,
        "email": [f"{f.lower()}.{l.lower()}{i}@example.com" for f, l, i in zip(first, last, ids)],
        "segment": segments,
        "region": regions,
        "city": cities,
        "signup_date": signup,
    })
    return df


def generate_products(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    pid = 1
    shares = np.array([v[0] for v in CATEGORIES.values()])
    counts = np.maximum(8, np.round(shares / shares.sum() * N_PRODUCTS).astype(int))
    for (cat, (_, (lo, hi), (cr_lo, cr_hi))), n in zip(CATEGORIES.items(), counts):
        # log-uniform prices: many cheap items, fewer expensive ones
        prices = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        for k in range(n):
            price = round(float(prices[k]) / 10) * 10 - 1  # e.g. 1499, 24999
            price = max(price, lo)
            cost = round(price * float(rng.uniform(cr_lo, cr_hi)), 2)
            sub = SUB_CATEGORIES[cat][k % len(SUB_CATEGORIES[cat])]
            rows.append({
                "product_id": pid,
                "product_name": f"{sub} {cat.split()[0]} {k + 1:02d}",
                "category": cat,
                "sub_category": sub,
                "unit_price": float(price),
                "unit_cost": cost,
            })
            pid += 1
    return pd.DataFrame(rows)


def _month_starts() -> list[date]:
    months, d = [], date(START_DATE.year, START_DATE.month, 1)
    while d <= END_DATE:
        months.append(d)
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return months


def _days_in_month(m: date) -> int:
    nxt = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
    return (nxt - m).days


def generate_orders(rng, customers: pd.DataFrame, products: pd.DataFrame):
    cust_ids = customers["customer_id"].to_numpy()
    cust_region = customers["region"].to_numpy()
    cust_signup = pd.to_datetime(customers["signup_date"]).dt.date.to_numpy()

    # Customer activity: lognormal; VIPs buy far more often.
    base_weight = rng.lognormal(0.0, 0.8, N_CUSTOMERS)
    base_weight[:N_VIP] *= 12
    vip_mask = np.zeros(N_CUSTOMERS, dtype=bool)
    vip_mask[:N_VIP] = True
    reducer_mask = np.zeros(N_CUSTOMERS, dtype=bool)
    reducer_mask[:N_VIP_REDUCERS] = True

    cats = list(CATEGORIES)
    base_cat_share = np.array([CATEGORIES[c][0] for c in cats])
    prod_by_cat = {c: products[products["category"] == c] for c in cats}
    prod_weights = {c: rng.lognormal(0, 0.7, len(prod_by_cat[c])) for c in cats}

    orders, items, payments = [], [], []
    order_id, item_id, pay_id = 1, 1, 1
    months = _month_starts()

    for mi, m in enumerate(months):
        trend = 1.0 + 0.15 * mi / (len(months) - 1)          # gentle growth
        n_orders = int(round(BASE_ORDERS_PER_MONTH * SEASONALITY[m.month] * trend
                             * rng.normal(1.0, 0.03)))
        month_end = m + timedelta(days=_days_in_month(m) - 1)

        # Customer weights for this month
        w = base_weight.copy()
        if m.year == 2025:
            w[reducer_mask] *= VIP_REDUCER_2025[m.month]
        eligible = np.array([s <= month_end for s in cust_signup])
        w = np.where(eligible, w, 0.0)
        w = w / w.sum()

        # Category mix for this month
        cat_share = base_cat_share.copy()
        if m.year == 2025:
            cat_share[cats.index("Electronics")] *= ELECTRONICS_DECLINE_2025[m.month]
        cat_share = cat_share / cat_share.sum()

        chosen = rng.choice(N_CUSTOMERS, n_orders, p=w)
        days = rng.integers(0, _days_in_month(m), n_orders)
        festive = m.month in (10, 11)

        for idx, dday in zip(chosen, days):
            odate = m + timedelta(days=int(dday))
            if odate < cust_signup[idx]:
                odate = cust_signup[idx]
            is_vip = vip_mask[idx]
            n_items = 1 + int(rng.poisson(0.7 + (1.5 if is_vip else 0.0)))
            n_items = min(n_items, 8)

            item_rows, has_electronics = [], False
            for _ in range(n_items):
                cat = cats[int(rng.choice(len(cats), p=cat_share))]
                has_electronics |= cat == "Electronics"
                pool = prod_by_cat[cat]
                pw = prod_weights[cat] / prod_weights[cat].sum()
                prod = pool.iloc[int(rng.choice(len(pool), p=pw))]
                if cat in ("Electronics", "Furniture"):
                    qty = 1 + int(rng.random() < (0.35 if is_vip else 0.05))
                else:
                    qty = 1 + int(rng.poisson(0.5 + (1.5 if is_vip else 0.0)))
                disc_choices = [0, 0.05, 0.10, 0.15, 0.20, 0.25]
                disc_p = [0.15, 0.20, 0.25, 0.20, 0.12, 0.08] if festive \
                    else [0.55, 0.20, 0.15, 0.07, 0.03, 0.00]
                disc = float(rng.choice(disc_choices, p=disc_p))
                item_rows.append((int(prod["product_id"]), qty, float(prod["unit_price"]),
                                  float(prod["unit_cost"]), disc))

            # Status (Electronics returns more often)
            p_ret = 0.09 if has_electronics else 0.04
            r = rng.random()
            status = "Cancelled" if r < 0.07 else ("Returned" if r < 0.07 + p_ret else "Completed")

            ship_region = cust_region[idx] if rng.random() < 0.92 \
                else str(rng.choice(list(REGIONS)))
            orders.append((order_id, int(cust_ids[idx]), odate, status, ship_region))

            total = 0.0
            for pid, qty, price, cost, disc in item_rows:
                items.append((item_id, order_id, pid, qty, price, cost, disc))
                total += qty * price * (1 - disc)
                item_id += 1

            method = str(rng.choice(PAYMENT_METHODS, p=PAYMENT_PROBS))
            pstatus = {"Completed": "Paid", "Returned": "Refunded", "Cancelled": "Cancelled"}[status]
            pdate = odate + timedelta(days=int(rng.integers(0, 3)))
            payments.append((pay_id, order_id, method, round(total, 2), pdate, pstatus))
            pay_id += 1
            order_id += 1

    orders_df = pd.DataFrame(orders, columns=["order_id", "customer_id", "order_date", "status", "region"])
    items_df = pd.DataFrame(items, columns=["order_item_id", "order_id", "product_id", "quantity",
                                            "unit_price", "unit_cost", "discount_pct"])
    pay_df = pd.DataFrame(payments, columns=["payment_id", "order_id", "payment_method", "amount",
                                             "payment_date", "payment_status"])
    return orders_df, items_df, pay_df


def inject_data_quality_issues(rng, customers: pd.DataFrame) -> pd.DataFrame:
    """Make the raw customers file realistically messy: missing cities and exact duplicate rows.
    setup_database.py cleans these up (so cleaning logic is demonstrable)."""
    df = customers.copy()
    missing_idx = rng.choice(df.index, 25, replace=False)
    df.loc[missing_idx, "city"] = None
    dupes = df.sample(15, random_state=SEED)
    return pd.concat([df, dupes], ignore_index=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rng = np.random.default_rng(SEED)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    customers = generate_customers(rng)
    products = generate_products(rng)
    orders, items, payments = generate_orders(rng, customers, products)
    customers_raw = inject_data_quality_issues(rng, customers)

    for name, df in [("customers", customers_raw), ("products", products), ("orders", orders),
                     ("order_items", items), ("payments", payments)]:
        df.to_csv(RAW_DIR / f"{name}.csv", index=False)
        logger.info("Wrote %-12s %6d rows -> data/raw/%s.csv", name, len(df), name)


if __name__ == "__main__":
    main()