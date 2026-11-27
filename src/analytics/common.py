"""
src/analytics/common.py

Small shared helpers for the analytics modules: errors, month arithmetic,
safe maths, money/percent formatting and JSON-friendly conversion.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, is_dataclass

import numpy as np
import pandas as pd


class AnalyticsError(Exception):
    """Raised for bad input (missing columns, unknown month, ...). Message is user-safe."""


_MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def check_month(month: str) -> str:
    if not isinstance(month, str) or not _MONTH_RE.match(month):
        raise AnalyticsError(f"Invalid month '{month}'. Use the format YYYY-MM, for example 2025-03.")
    return month


def shift_month(month: str, k: int) -> str:
    return str(pd.Period(check_month(month), freq="M") + k)


def prev_month(month: str) -> str:
    return shift_month(month, -1)


def require_columns(df: pd.DataFrame, columns: list[str], name: str = "data") -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise AnalyticsError(f"The {name} table is missing required column(s): {', '.join(missing)}.")


def safe_div(a, b):
    """a / b, or None when b is zero/missing."""
    if b is None or a is None or pd.isna(a) or pd.isna(b) or float(b) == 0.0:
        return None
    return float(a) / float(b)


def pct_change(new, old):
    """Percentage change from old to new; None when old is zero/missing (undefined)."""
    if old is None or new is None or pd.isna(old) or pd.isna(new) or float(old) == 0.0:
        return None
    return (float(new) - float(old)) / abs(float(old)) * 100.0


def fmt_money(x) -> str:
    """Indian-style short money format: ₹59.77 lakh, ₹1.70 crore, ₹14,868."""
    if x is None or pd.isna(x):
        return "n/a"
    sign = "-" if x < 0 else ""
    v = abs(float(x))
    if v >= 1e7:
        return f"{sign}₹{v / 1e7:.2f} crore"
    if v >= 1e5:
        return f"{sign}₹{v / 1e5:.2f} lakh"
    return f"{sign}₹{v:,.0f}"


def fmt_pct(x, signed: bool = True) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:+.1f}%" if signed else f"{x:.1f}%"


def to_py(obj):
    """Recursively convert numpy/pandas objects into plain Python so results can be json.dumps'd."""
    if isinstance(obj, pd.DataFrame):
        return [to_py(r) for r in obj.to_dict("records")]
    if isinstance(obj, pd.Series):
        return to_py(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_py(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if obj is pd.NaT or obj is pd.NA:
        return None
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_py(asdict(obj))
    return obj