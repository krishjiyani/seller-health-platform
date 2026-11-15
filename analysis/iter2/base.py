"""Shared base for the iteration 2 feasibility tests: pipeline orders and seller panel plus corrected labels, cached.

Corrected labels (see PROGRESS.md S1 to S4):
* late_date: delivered after the promised calendar date (the promise has no time of day).
* late_or_lost: late_date, or handed to the carrier and never delivered (every such promise passed before the data ends).
* late_dispatch_full: late handover, or never handed over (status processing or invoiced, ship limit passed).
* clean: promised date on or before 29 Aug 2018, so the review survey (promise + 2 at the latest) fell inside the data.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

CACHE = ROOT / "data" / "interim"
CLEAN_PROMISE_MAX = pd.Timestamp("2018-08-29")


def load():
    CACHE.mkdir(parents=True, exist_ok=True)
    fo, fp = CACHE / "orders_v2.parquet", CACHE / "panel_v2.parquet"
    if fo.exists() and fp.exists():
        o, panel = pd.read_parquet(fo), pd.read_parquet(fp)
        for c in ["category", "customer_state", "seller_state", "pay_type"]:
            o[c] = o[c].astype("category")
        return o, panel
    from pipeline import build, models
    con = build.load_raw()
    orders = build.build_orders(con)
    # The pipeline adopted the corrected labels in iteration 2. These tests compare against iteration 1, so restore the
    # iteration 1 definitions here; the corrected ones are rebuilt below under their own names.
    orders["late_dispatch"] = (orders.ts_carrier > orders.ship_limit).astype(float).where(orders.ts_carrier.notna() & orders.ship_limit.notna())
    orders["late_delivery"] = (orders.ts_delivered > orders.ts_estimated).astype(float).where(orders.ts_delivered.notna())
    orders["unjust_review"] = (orders.low_review == 1) & (orders.late_dispatch == 0) & (orders.late_delivery == 1)
    panel = build.build_seller_months(orders)
    o = models.add_order_features(orders, panel)
    o["late_date"] = (o.ts_delivered.dt.normalize() > o.ts_estimated).astype(float).where(o.ts_delivered.notna())
    lost = o.ts_carrier.notna() & o.ts_delivered.isna() & ~o.failed
    o["lost"] = lost
    o["late_or_lost"] = o.late_date.where(~lost, 1.0)
    never = o.ts_carrier.isna() & ~o.failed & o.ship_limit.notna()
    o["never_dispatched"] = never
    o["late_dispatch_full"] = o.late_dispatch.where(~never, 1.0)
    o["clean"] = o.ts_estimated <= CLEAN_PROMISE_MAX
    o["lane"] = (o.seller_state.astype(str) + ">" + o.customer_state.astype(str)).astype("category")
    o["actual_d"] = (o.ts_delivered - o.ts_purchase).dt.total_seconds() / 86400
    o["transit_d"] = (o.ts_delivered - o.ts_carrier).dt.total_seconds() / 86400
    o["handover_d"] = (o.ts_carrier - o.ts_purchase).dt.total_seconds() / 86400
    o.to_parquet(fo, index=False)
    panel.to_parquet(fp, index=False)
    return o, panel


def lane_day_stats(o: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Live carrier signals per (lane, day) using only parcels delivered or handed over BEFORE that day starts.

    lane_transit_med_30: median transit of parcels delivered on the lane in the 30 days before the day.
    lane_late_share_30: share of those deliveries that missed the promise date.
    dest_backlog: parcels to the destination state handed over 8 to 40 days earlier and still undelivered at day start,
                  divided by parcels handed over to that state in the 30 days before (a live congestion signal).
    plat_transit_med_7: platform median transit of deliveries in the 7 days before the day."""
    d = o[o.ts_delivered.notna() & o.ts_carrier.notna()][["lane", "customer_state", "ts_delivered", "transit_d", "late_date"]].copy()
    d["dday"] = d.ts_delivered.dt.normalize()
    rows = []
    for day in days:
        w = d[(d.dday < day) & (d.dday >= day - pd.Timedelta(days=30))]
        g = w.groupby("lane", observed=True).agg(lane_transit_med_30=("transit_d", "median"), lane_late_share_30=("late_date", "mean"),
                                                  lane_n_30=("transit_d", "size"))
        g["day"] = day
        g["plat_transit_med_7"] = d[(d.dday < day) & (d.dday >= day - pd.Timedelta(days=7))].transit_d.median()
        rows.append(g.reset_index())
    lane = pd.concat(rows, ignore_index=True)
    h = o[o.ts_carrier.notna()][["customer_state", "ts_carrier", "ts_delivered"]]
    brow = []
    for day in days:
        handed = h[(h.ts_carrier < day - pd.Timedelta(days=8)) & (h.ts_carrier >= day - pd.Timedelta(days=40))]
        open_old = handed[handed.ts_delivered.isna() | (handed.ts_delivered >= day)].groupby("customer_state", observed=True).size()
        recent = h[(h.ts_carrier < day) & (h.ts_carrier >= day - pd.Timedelta(days=30))].groupby("customer_state", observed=True).size()
        b = (open_old.reindex(recent.index).fillna(0) / recent.clip(lower=1)).rename("dest_backlog").reset_index()
        b["day"] = day
        brow.append(b)
    back = pd.concat(brow, ignore_index=True)
    return lane, back


def seller_day_stats(o: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Live seller signals per (seller, day), using only events before the day starts.

    s_open_now: orders bought in the last 30 days not yet handed over (includes orders later cancelled: their fate is unknown live).
    s_late_14: late share among handovers completed in the last 14 days.
    s_handover_med_30: median purchase to handover days among handovers completed in the last 30 days.
    s_surge: orders bought in the last 7 days divided by the seller's 7 day average over the previous 90 days."""
    x = o[o.seller_id.notna()][["seller_id", "ts_purchase", "ts_carrier", "late_dispatch", "handover_d"]].copy()
    sid = x.seller_id.astype("category")
    x["s"] = sid.cat.codes
    d0 = days.min() - pd.Timedelta(days=120)
    nd = (days.max() - d0).days + 2
    pday = ((x.ts_purchase.dt.normalize() - d0).dt.days).clip(0, nd - 1).values
    hday = ((x.ts_carrier.dt.normalize() - d0).dt.days)
    S = len(sid.cat.categories)

    def grid(day_idx, weights=None, mask=None):
        g = np.zeros((S, nd))
        m = np.ones(len(x), bool) if mask is None else mask
        np.add.at(g, (x.s.values[m], day_idx[m]), 1 if weights is None else weights[m])
        return g.cumsum(1)

    purch = grid(pday)
    hm = hday.notna().values & (hday.values >= 0)
    hidx = np.where(hm, hday.fillna(0).astype(int).clip(0, nd - 1).values, 0)
    hand = grid(hidx, mask=hm)
    late = grid(hidx, weights=x.late_dispatch.fillna(0).values, mask=hm)
    # open orders: +1 on purchase day, -1 on min(handover day, purchase + 30)
    end = np.where(hm, np.minimum(hidx, pday + 30), pday + 30).clip(0, nd - 1)
    diff = np.zeros((S, nd + 1))
    np.add.at(diff, (x.s.values, pday), 1)
    np.add.at(diff, (x.s.values, end), -1)
    openg = diff.cumsum(1)[:, :nd]
    out = []
    hsorted = x[hm].sort_values("ts_carrier")
    for day in days:
        j = (day - d0).days          # index of this day; use cumulative counts up to j - 1 (events before the day)
        f = pd.DataFrame({"seller_id": sid.cat.categories})
        f["day"] = day
        f["s_open_now"] = openg[:, j - 1]
        n14 = hand[:, j - 1] - hand[:, j - 15]
        f["s_late_14"] = np.where(n14 > 0, (late[:, j - 1] - late[:, j - 15]) / np.maximum(n14, 1), np.nan)
        f["s_hand_n_14"] = n14
        o7 = purch[:, j - 1] - purch[:, j - 8]
        o90 = purch[:, j - 8] - purch[:, j - 98]
        f["s_surge"] = (o7 + 1) / (o90 / 13 + 1)
        w = hsorted[(hsorted.ts_carrier < day) & (hsorted.ts_carrier >= day - pd.Timedelta(days=30))]
        med = w.groupby("seller_id").handover_d.median()
        f["s_handover_med_30"] = f.seller_id.map(med)
        out.append(f)
    return pd.concat(out, ignore_index=True)
