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
PARTIAL_FROM = pd.Timestamp("2018-08-15")   # orders after this are a fast delivery only sample (reviews stop on 31 Aug 2018)
HIGH_SILENCE = 0.50
LOST_FLAG_D = 20
SURVEY_AFTER_PROMISE_D = 2
TIER_NAMES = {2: "High", 1: "Elevated", 0: "Normal"}
TAG_COLS = {"Not received": "t_not_received", "Late or slow delivery": "t_late", "Missing item": "t_missing",
            "Wrong item or not as described": "t_wrong", "Defect or quality": "t_defect", "Refund or return": "t_refund"}
TAG_PRECISION = {"Not received": 0.84, "Late or slow delivery": 0.96, "Missing item": 0.80, "Wrong item or not as described": 0.92,
                 "Defect or quality": 0.92, "Refund or return": 1.00}
SHAPES = ["1 item", "2+ items, 1 seller", "2+ sellers"]
OUTCOMES = ["On time", "Late", "Lost in transit", "Never dispatched", "Cancelled or unavailable"]


@st.cache_data
def orders() -> pd.DataFrame:
    o = pd.read_parquet(APP_DATA / "orders.parquet")
    b = lambda s: s.fillna(False).to_numpy(dtype=bool)
    shape = np.select([b(o.n_sellers >= 2), b(o.n_items >= 2)], ["2+ sellers", "2+ items, 1 seller"], "1 item")
    o["order_shape"] = pd.Series(shape, index=o.index).where(o.n_items.notna())
    outcome = np.select([b(o.failed), b(o.never_dispatched), b(o.lost), b(o.late_delivery == 1), b(o.late_delivery == 0)],
                        ["Cancelled or unavailable", "Never dispatched", "Lost in transit", "Late", "On time"], "")
    o["outcome"] = pd.Series(outcome, index=o.index).replace("", None)
    return o


@st.cache_data
def daily(window: str) -> pd.DataFrame:
    """Daily snapshot risk: one probability and tier per order per day it was inside the window (see pipeline/models.py)."""
    d = pd.read_parquet(APP_DATA / f"daily_{window}.parquet")
    d["day"] = pd.to_datetime(d.day)
    return d


def _risk_on(window: str, T: pd.Timestamp) -> pd.DataFrame:
    d = daily(window)
    return d[d.day == T.normalize()].set_index("order_id")[["p", "tier"]]


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
def sellers() -> pd.DataFrame:
    return pd.read_parquet(APP_DATA / "sellers.parquet")


@st.cache_data
def metrics() -> dict:
    return json.loads((APP_DATA / "metrics.json").read_text())


def short_id(s: pd.Series) -> pd.Series:
    return s.str[:8]


def panel_month_for(T: pd.Timestamp) -> str:
    """Latest month end on or before T (the seller panel is computed at month ends)."""
    M = T.to_period("M")
    return str(M if T >= M.to_timestamp(how="end").normalize() else M - 1)


def tier_hit_rates(window: str) -> dict:
    """Measured late share per tier, averaged over the test months from Sep 2017 (walk forward, clean cohort)."""
    m = pd.DataFrame(metrics()[f"{window}_daily"])
    m = m[m.month >= "2017-09"]
    return {name: float(m[f"{name.lower()}_rate"].mean()) for name in TIER_NAMES.values()}


def dispatch_queue(T: pd.Timestamp) -> pd.DataFrame:
    """Orders bought by T and not yet handed to the carrier at T, with deadline within the last 14 days or ahead.

    Cancelled orders are excluded because the data does not record when they were cancelled (stated on the Evidence page)."""
    o = orders()
    q = o[(o.ts_purchase <= T) & (o.ship_limit.notna()) & (o.ts_carrier.isna() | (o.ts_carrier > T)) & ~o.failed
          & (o.ship_limit >= T - pd.Timedelta(days=14))].copy()
    q["days_to_deadline"] = (q.ship_limit - T).dt.total_seconds() / 86400
    q["overdue"] = q.days_to_deadline < 0
    r = _risk_on("dispatch", T)
    q["p"] = q.order_id.map(r.p)
    q["tier"] = np.where(q.overdue, "Overdue", q.order_id.map(r.tier).map(TIER_NAMES).fillna("Not scored"))
    q["flag"] = q.overdue | (q.tier == "High")
    return q


def carrier_queue(T: pd.Timestamp) -> pd.DataFrame:
    """Parcels with the carrier at T whose review survey has not gone out yet, plus parcels long enough in transit to be lost.

    The survey goes out the day after delivery, or 2 days after the promised date if the parcel has not arrived. Before the
    promise the model tier applies; after it the parcel is late for certain and the KAM has until the survey date."""
    o = orders()
    q = o[(o.ts_carrier <= T) & (o.ts_delivered.isna() | (o.ts_delivered > T)) & ~o.failed].copy()
    q["survey_date"] = q.ts_estimated + pd.Timedelta(days=SURVEY_AFTER_PROMISE_D)
    q["days_to_survey"] = (q.survey_date - T.normalize()).dt.days
    q["days_in_transit"] = (T - q.ts_carrier).dt.total_seconds() / 86400
    q["days_to_promise"] = (q.ts_estimated - T.normalize()).dt.days
    q = q[(q.days_to_survey >= 0) | ((q.days_in_transit >= LOST_FLAG_D) & (q.days_in_transit < 60))].copy()
    q["before_promise"] = q.ts_estimated >= T
    q["promise_broken"] = ~q.before_promise & (q.days_to_survey >= 0)
    # Possibly lost: 20+ days with the carrier and already past the promise (parcels before their promise are in the tiers).
    q["possibly_lost"] = (q.days_in_transit >= LOST_FLAG_D) & ~q.before_promise
    q["seller_was_late"] = q.dispatch_delay_d > 0
    r = _risk_on("carrier", T)
    q["p"] = q.order_id.map(r.p).where(q.before_promise)
    q["tier"] = np.where(q.promise_broken, "Promise missed",
                         np.where(q.before_promise, q.order_id.map(r.tier).map(TIER_NAMES).fillna("Not scored"), "Survey sent"))
    q["flag"] = (q.tier == "High") | q.promise_broken
    return q


@st.cache_data
def lost_curve() -> pd.DataFrame:
    """Share of parcels still with the carrier d days after handover, and past their promised date, that never arrive
    (clean cohort, measured)."""
    o = orders()
    h = o[o.ts_carrier.notna() & ~o.failed & o.clean & (o.ts_delivered.notna() | o.lost)]
    transit = (h.ts_delivered - h.ts_carrier).dt.total_seconds() / 86400
    rows = []
    for d in [20, 25, 30, 40, 45, 60]:
        day_d = h.ts_carrier + pd.Timedelta(days=d)
        still = h[(transit.isna() | (transit > d)) & (day_d > h.ts_estimated + pd.Timedelta(days=1))]
        rows.append(dict(day=d, in_transit=len(still), never_arrive=float(still.lost.mean())))
    return pd.DataFrame(rows)


def lost_chance(days: pd.Series) -> pd.Series:
    c = lost_curve().set_index("day").never_arrive
    return days.map(lambda x: c.iloc[np.searchsorted(c.index, x, side="right") - 1] if x >= c.index[0] else np.nan)


def silence_list(T: pd.Timestamp) -> pd.DataFrame:
    sm = seller_months()
    s = sm[(sm.month == panel_month_for(T)) & sm.p_silent_1.notna()].copy()
    s["window"] = np.select([s.p_silent_1 >= HIGH_SILENCE, s.p_silent_3 >= HIGH_SILENCE], ["Act now", "Window open"], "Stable")
    s["gmv_at_stake"] = s.g_90 * s.p_silent_3
    s["basis"] = np.where(s.big_seller.fillna(False), "Days since last order", "Model")
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


def lead_decay() -> pd.DataFrame:
    """Share of leads that eventually sign, given they have not signed by day d (MQLs first contacted Dec 2017 to May 2018)."""
    q = mql()
    q = q[q.lead_cohort]
    days = ((q.won_date - q.first_contact_date).dt.total_seconds() / 86400).clip(lower=0)
    rows = []
    for d in [0, 7, 14, 21, 30, 45, 60, 90]:
        alive = q[~(days < d)]
        rows.append(dict(day=d, open_leads=len(alive), p_win=float(alive.won.mean())))
    return pd.DataFrame(rows)


def ramp_rule() -> pd.DataFrame:
    return pd.DataFrame(metrics()["ramp_rule"])


def ramp_queue(T: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    """Sellers 30 to 60 days after their first sale with at most 2 orders in their first 30 days, and the count still in days 0 to 30."""
    s = sellers()
    o = orders()
    age = (T - s.first_sale).dt.total_seconds() / 86400
    young = s[(age >= 0) & (age < 30)]
    q = s[(age >= 30) & (age <= 60)].copy()
    q["days_since_first_sale"] = age[q.index].astype(int)
    oo = o[o.seller_id.isin(q.seller_id) & (o.ts_purchase <= T)].merge(q[["seller_id", "first_sale"]], on="seller_id")
    a = (oo.ts_purchase - oo.first_sale).dt.total_seconds() / 86400
    q["orders_30"] = q.seller_id.map(oo[a <= 30].groupby("seller_id").size()).fillna(0).astype(int)
    rule = ramp_rule().set_index("orders_30").established_rate
    q["p_established"] = q.orders_30.clip(upper=6).map(rule)
    later = o[o.seller_id.isin(q.seller_id)].merge(q[["seller_id", "first_sale"]], on="seller_id")
    la = (later.ts_purchase - later.first_sale).dt.total_seconds() / 86400
    q["orders_31_180"] = q.seller_id.map(later[(la > 30) & (la <= 180)].groupby("seller_id").size()).fillna(0).astype(int)
    q["observed_180"] = q.first_sale + pd.Timedelta(days=180) <= LAST_ASOF
    return q[q.orders_30 <= 2].sort_values("days_since_first_sale"), len(young)


def shape_rates(on_time_only: bool = True) -> pd.DataFrame:
    """Bad review rate by order shape (clean cohort, reviewed orders)."""
    o = orders()
    r = o[o.low_review.notna() & o.clean & o.order_shape.notna()]
    if on_time_only:
        r = r[r.outcome == "On time"]
    g = r.groupby("order_shape").low_review.agg(["mean", "size"]).reindex(SHAPES)
    g["share_of_bad"] = r.groupby("order_shape").low_review.sum().reindex(SHAPES) / r.low_review.sum()
    return g.rename(columns={"mean": "bad_rate", "size": "orders"})
