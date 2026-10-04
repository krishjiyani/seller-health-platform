"""Loads the precomputed tables in data/app and answers "what did the system know on date T" questions.

Nothing here trains a model. Every probability was produced by the pipeline using only data available
before the month it scores (see pipeline/models.py)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

APP_DATA = Path(__file__).resolve().parents[2] / "data" / "app"
FIRST_ASOF = pd.Timestamp("2017-06-01")
LAST_ASOF = pd.Timestamp("2018-08-31")
DEFAULT_ASOF = pd.Timestamp("2018-03-15")
HIGH_DISPATCH, HIGH_CARRIER, HIGH_SILENCE = 0.25, 0.30, 0.50


@st.cache_data
def orders() -> pd.DataFrame:
    return pd.read_parquet(APP_DATA / "orders.parquet")


@st.cache_data
def daily(window: str) -> pd.DataFrame:
    """Daily snapshot risk: one probability per order per day it was inside the window (see pipeline/models.py)."""
    d = pd.read_parquet(APP_DATA / f"daily_{window}.parquet")
    d["day"] = pd.to_datetime(d.day)
    return d


def _risk_on(window: str, T: pd.Timestamp) -> pd.Series:
    d = daily(window)
    return d[d.day == T.normalize()].set_index("order_id").p


@st.cache_data
def seller_months() -> pd.DataFrame:
    return pd.read_parquet(APP_DATA / "seller_months.parquet")


@st.cache_data
def deals() -> pd.DataFrame:
    return pd.read_parquet(APP_DATA / "deals.parquet")


@st.cache_data
def mql() -> pd.DataFrame:
    return pd.read_parquet(APP_DATA / "mql.parquet")


@st.cache_data
def metrics() -> dict:
    return json.loads((APP_DATA / "metrics.json").read_text())


def short_id(s: pd.Series) -> pd.Series:
    return s.str[:8]


def panel_month_for(T: pd.Timestamp) -> str:
    """Latest month end on or before T (the seller panel is computed at month ends)."""
    M = T.to_period("M")
    return str(M if T >= M.to_timestamp(how="end").normalize() else M - 1)


def dispatch_queue(T: pd.Timestamp) -> pd.DataFrame:
    """Orders bought by T and not yet handed to the carrier at T, with deadline within the last 14 days or ahead.

    Cancelled orders are excluded because the data does not record when they were cancelled (stated on the Evidence page)."""
    o = orders()
    q = o[(o.ts_purchase <= T) & (o.ship_limit.notna()) & (o.ts_carrier.isna() | (o.ts_carrier > T)) & ~o.failed
          & (o.ship_limit >= T - pd.Timedelta(days=14))].copy()
    q["days_to_deadline"] = (q.ship_limit - T).dt.total_seconds() / 86400
    q["overdue"] = q.days_to_deadline < 0
    q["risk"] = np.where(q.overdue, 1.0, q.order_id.map(_risk_on("dispatch", T)))
    q["flag"] = q.overdue | (q.risk >= HIGH_DISPATCH)
    return q


def carrier_queue(T: pd.Timestamp) -> pd.DataFrame:
    """Orders with the carrier at T whose promised date has not passed yet: the window before the review survey fires."""
    o = orders()
    q = o[(o.ts_carrier <= T) & (o.ts_delivered.isna() | (o.ts_delivered > T)) & (o.ts_estimated >= T) & ~o.failed].copy()
    q["days_to_promise"] = (q.ts_estimated - T).dt.total_seconds() / 86400
    q["seller_was_late"] = q.dispatch_delay_d > 0
    q["risk"] = q.order_id.map(_risk_on("carrier", T))
    q["flag"] = q.risk >= HIGH_CARRIER
    return q


def silence_list(T: pd.Timestamp) -> pd.DataFrame:
    sm = seller_months()
    s = sm[(sm.month == panel_month_for(T)) & sm.p_silent_1.notna()].copy()
    s["window"] = np.select([s.p_silent_1 >= HIGH_SILENCE, s.p_silent_3 >= HIGH_SILENCE], ["Act now", "Window open"], "Stable")
    s["gmv_at_stake"] = s.g_90 * s.p_silent_3
    return s


SIGNAL_RULES = [
    ("recency_d", lambda v: v > 30, lambda v: f"No order in {int(v)} days"),
    ("momentum", lambda v: v < -0.3, lambda v: "Orders slowing (last 30 days vs 60 to 90 days ago)"),
    ("gmv_trend", lambda v: v < -0.3, lambda v: "Sales down vs previous 90 days"),
    ("late_dispatch_180", lambda v: v > 0.15, lambda v: f"Late dispatch on {v:.0%} of recent orders"),
    ("overdue_now", lambda v: v > 0, lambda v: f"{int(v)} order(s) past dispatch deadline"),
    ("low_review_share_180", lambda v: v > 0.25, lambda v: f"{v:.0%} of recent reviews are 1 or 2 stars"),
    ("fail_rate_90", lambda v: v > 0.05, lambda v: f"{v:.0%} of recent orders cancelled or unavailable"),
]


def signals(row) -> list[str]:
    out = []
    for col, test, text in SIGNAL_RULES:
        v = row.get(col)
        if pd.notna(v) and test(v):
            out.append(text(v))
    return out


def onboarding_curve() -> pd.DataFrame:
    """P(first sale by day 90 | no sale by day d), from deals won by 2018-06-01 so every deal has 90 days of runway."""
    d = deals()
    el = d[d.won_date <= "2018-06-01"]
    lag = el.days_to_first_sale
    rows = []
    for day in range(0, 91, 5):
        alive = el[~(lag < day)]
        rows.append(dict(day=day, still_unsold=len(alive), p_sell_by_90=float(lag[alive.index].between(day, 90).mean())))
    return pd.DataFrame(rows)


def stuck_sellers(T: pd.Timestamp) -> pd.DataFrame:
    d = deals()
    s = d[(d.won_date <= T) & (d.first_sale.isna() | (d.first_sale > T))].copy()
    s["days_since_won"] = (T - s.won_date).dt.days
    curve = onboarding_curve().set_index("day").p_sell_by_90
    s["p_sell_by_90"] = s.days_since_won.clip(upper=90).map(lambda x: curve.iloc[np.searchsorted(curve.index, x, side="right") - 1])
    s.loc[s.days_since_won > 90, "p_sell_by_90"] = np.nan
    s["stage"] = pd.cut(s.days_since_won, [-1, 14, 44, 90, 10_000],
                        labels=["Welcome (day 0 to 14)", "Setup call (day 15 to 44)", "Escalate (day 45 to 90)", "Likely lost (90+)"])
    return s
