"""Data sanity v2, part 3: survey timing rule, end of data selection, promise padding over time, lost parcels.

Run: python -m analysis.iter2.sanity_03_mechanics"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from pipeline import build

con = build.load_raw()
o = build.build_orders(con)
o["deliv_day"] = o.ts_delivered.dt.normalize()
o["late_date"] = (o.deliv_day > o.ts_estimated).astype(float).where(o.ts_delivered.notna())
DATA_CUT = pd.Timestamp("2018-10-17")   # last delivered timestamp in the raw data

# 1. Survey timing: is review_creation_date = min(delivery day + 1, promise + k)?
print("=== 1. Survey timing rule ===")
x = o[o.ts_review.notna() & o.ts_delivered.notna()].copy()
for k in (1, 2, 3):
    rule = np.minimum(x.deliv_day + pd.Timedelta(days=1), x.ts_estimated + pd.Timedelta(days=k))
    print(f"review date == min(delivery+1, promise+{k}): {(x.ts_review == rule).mean():.3f} | within 1 day: "
          f"{((x.ts_review - rule).dt.days.abs() <= 1).mean():.3f}")
rule = np.minimum(x.deliv_day + pd.Timedelta(days=1), x.ts_estimated + pd.Timedelta(days=2))
x["rule_gap"] = (x.ts_review - rule).dt.days
print("gap vs rule (days), share:", x.rule_gap.clip(-5, 5).value_counts(normalize=True).sort_index().round(3).to_dict())
for yr in ["2017", "2018"]:
    y = x[x.ts_purchase.dt.year == int(yr)]
    print(f"{yr}: rule exact match {(y.rule_gap == 0).mean():.3f}")
u = o[o.ts_delivered.isna() & o.ts_review.notna() & o.ts_estimated.notna()]
print("never delivered: review minus promise (days) share:", (u.ts_review - u.ts_estimated).dt.days.clip(-10, 10).value_counts(normalize=True).sort_index().round(3).head(25).to_dict())

# 2. End of data: are reviews capped at 31 Aug 2018, and does that select which late August orders exist?
print("\n=== 2. End of data selection ===")
a = o[o.ts_purchase >= "2018-07-01"].copy()
a["wk"] = a.ts_purchase.dt.to_period("W-SUN")
g = a.groupby("wk").agg(orders=("order_id", "size"), delivered_after_aug=("ts_delivered", lambda s: (s > "2018-08-31 23:59").mean()),
                        review_before_delivery=("ts_review", "size"))
a["rev_before_deliv"] = a.ts_review < a.deliv_day
g["rev_before_deliv"] = a.groupby("wk").rev_before_deliv.mean()
g["late_date"] = a.groupby("wk").late_date.mean()
g["median_promise_d"] = a.groupby("wk").promise_d.median()
print(g.drop(columns="review_before_delivery").round(3).to_string())
late_aug = o[(o.ts_purchase >= "2018-08-15")]
print(f"orders bought 15 to 31 Aug 2018: {len(late_aug):,}; delivered after 31 Aug {(late_aug.ts_delivered > '2018-08-31 23:59').mean():.3f}; "
      f"of those, review dated before delivery {(late_aug[late_aug.ts_delivered > '2018-08-31 23:59'].ts_review < late_aug[late_aug.ts_delivered > '2018-08-31 23:59'].deliv_day).mean():.3f}")

# 3. Promise padding and late rate by month.
print("\n=== 3. Promise length, transit and late rate by purchase month ===")
o["m"] = o.ts_purchase.dt.to_period("M")
o["transit_d"] = (o.ts_delivered - o.ts_carrier).dt.total_seconds() / 86400
o["actual_d"] = (o.ts_delivered - o.ts_purchase).dt.total_seconds() / 86400
t = o.groupby("m").agg(orders=("order_id", "size"), promise_med=("promise_d", "median"), actual_med=("actual_d", "median"),
                       transit_med=("transit_d", "median"), late_ts=("late_delivery", "mean"), late_date=("late_date", "mean"),
                       late_dispatch=("late_dispatch", "mean"), low_review=("low_review", "mean"))
print(t.round(3).to_string())

# 4. Lost parcels and never dispatched orders: outcomes iteration 1 treats as unobservable.
print("\n=== 4. Orders with no outcome in iteration 1 labels ===")
lost = o[o.ts_carrier.notna() & o.ts_delivered.isna() & ~o.failed]
print(f"handed over, never delivered, not cancelled: {len(lost):,}; promise passed before data cut {(lost.ts_estimated < DATA_CUT).mean():.3f}; "
      f"low review {lost.low_review.mean():.3f}; status {lost.status.value_counts().to_dict()}")
nd = o[o.ts_carrier.isna() & ~o.failed & o.ship_limit.notna()]
print(f"never handed over, not cancelled: {len(nd):,}; ship limit passed before data cut {(nd.ship_limit < DATA_CUT).mean():.3f}; "
      f"low review {nd.low_review.mean():.3f}; status {nd.status.value_counts().to_dict()}")
fl = o[o.failed]
print(f"cancelled or unavailable: {len(fl):,}; low review {fl.low_review.mean():.3f}")
base_pos = (o.late_date == 1).sum()
print(f"late by date rule (delivered): {int(base_pos):,}; adding lost parcels past promise would add {int((lost.ts_estimated < DATA_CUT).sum()):,} positives "
      f"({(lost.ts_estimated < DATA_CUT).sum() / base_pos:.1%} more)")
disp_pos = (o.late_dispatch == 1).sum()
print(f"late dispatch (handed over): {int(disp_pos):,}; never handed over past ship limit would add {int((nd.ship_limit < DATA_CUT).sum()):,} "
      f"({(nd.ship_limit < DATA_CUT).sum() / disp_pos:.1%} more)")
print("lost parcels by purchase month:", lost.m.value_counts().sort_index().to_dict())

# 5. Deal sellers: any orders before the deal was won?
print("\n=== 5. Deal sellers selling before the won date ===")
deals = con.execute("select seller_id, won_date from deals").df()
first = con.execute("select i.seller_id, min(o.order_purchase_timestamp) first_any from items i join orders o using(order_id) group by 1").df()
dd = deals.merge(first, on="seller_id", how="left")
print(f"deal sellers with any order: {dd.first_any.notna().sum()}; first order before won date: {(dd.first_any < dd.won_date).sum()}")
print("days to first sale quantiles:", ((dd.first_any - dd.won_date).dt.days).describe().round(1).to_dict())
