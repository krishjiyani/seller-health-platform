"""Data sanity v2, part 2: how label definitions and data quirks move the iteration 1 headline numbers.

Uses the pipeline's own build_orders so the comparison is like for like. Run: python -m analysis.iter2.sanity_02_definitions"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from pipeline import build

con = build.load_raw()
o = build.build_orders(con)
print(f"orders in window: {len(o):,}")

# A. Late delivery: timestamp rule (iteration 1) vs calendar date rule (promise is a date with no time).
o["late_delivery_date"] = (o.ts_delivered.dt.normalize() > o.ts_estimated).astype(float).where(o.ts_delivered.notna())
d = o[o.late_delivery.notna()]
print("\n=== A. Late delivery definition ===")
print(f"timestamp rule {d.late_delivery.mean():.4f} | date rule {d.late_delivery_date.mean():.4f} | "
      f"flipped to on time {int((d.late_delivery == 1).sum() - (d.late_delivery_date == 1).sum()):,}")
for name, col in [("timestamp rule", "late_delivery"), ("date rule", "late_delivery_date")]:
    late = o[(o[col] == 1) & o.late_dispatch.notna()]
    unjust = ((o.low_review == 1) & (o.late_dispatch == 0) & (o[col] == 1)).sum()
    print(f"{name}: carrier fault share of late {1 - late.late_dispatch.mean():.3f} | late with late handover {late.late_dispatch.mean():.3f} | "
          f"low review on time {o[o[col] == 0].low_review.mean():.3f} vs late {o[o[col] == 1].low_review.mean():.3f} | unjust reviews {int(unjust):,}")
od = o[(o.late_delivery == 1) & (o.late_delivery_date == 0)]
print(f"delivered on promise day: n={len(od):,}, low review share {od.low_review.mean():.3f} (on time overall "
      f"{o[o.late_delivery_date == 0].low_review.mean():.3f}, late by date {o[o.late_delivery_date == 1].low_review.mean():.3f})")

# B. Review timing on late orders (the survey timing claim).
print("\n=== B. Review timing ===")
for name, col in [("timestamp rule", "late_delivery"), ("date rule", "late_delivery_date")]:
    x = o[(o[col] == 1) & o.ts_review.notna()]
    before = (x.ts_review < x.ts_delivered.dt.normalize()).mean()
    same = (x.ts_review <= x.ts_delivered.dt.normalize()).mean()
    print(f"{name}: late orders with review {len(x):,} | review a full day+ before arrival {before:.3f} | same day or before {same:.3f}")
x = o[o.ts_review.notna() & o.ts_delivered.notna()]
gap = (x.ts_review - x.ts_delivered.dt.normalize()).dt.days
print("review minus delivery date, all delivered orders, quantiles (days):", gap.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict())
gap_p = (x.ts_review - x.ts_estimated).dt.days
late = x.late_delivery_date == 1
print("late orders: review minus PROMISE date quantiles:", gap_p[late].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict())
print("on time orders: review minus DELIVERY date quantiles:", gap[~late].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict())
und = o[o.ts_delivered.isna() & o.ts_review.notna()]
print(f"reviews on never delivered orders: {len(und):,}, low share {und.low_review.mean():.3f}")

# C. Timestamp anomalies inside the modelling window.
print("\n=== C. Timestamp anomalies (window orders) ===")
a = {
    "carrier before purchase": (o.ts_carrier < o.ts_purchase).sum(),
    "carrier before approval": (o.ts_carrier < o.ts_approved).sum(),
    "delivered before carrier": (o.ts_delivered < o.ts_carrier).sum(),
    "transit over 60 days": ((o.ts_delivered - o.ts_carrier).dt.days > 60).sum(),
    "dispatch more than 30 days late": (o.dispatch_delay_d > 30).sum(),
    "ship limit over 30 days after purchase": (o.limit_d > 30).sum(),
}
for k, v in a.items():
    print(f"{k}: {int(v):,}")
cb = o[o.ts_carrier < o.ts_purchase]
print("carrier before purchase, by month:", cb.ts_purchase.dt.to_period("M").value_counts().sort_index().to_dict())
print("carrier before purchase, late_dispatch values:", cb.late_dispatch.value_counts().to_dict())

# D. Right censoring at the end of the data.
print("\n=== D. Censoring at data end (31 Aug 2018; raw data runs to Oct 2018) ===")
for m in ["2018-06", "2018-07", "2018-08"]:
    x = o[o.ts_purchase.dt.to_period("M") == m]
    print(f"{m}: orders {len(x):,} | no handover {x.ts_carrier.isna().mean():.3f} | not delivered {x.ts_delivered.isna().mean():.3f} | "
          f"no review {x.review_score.isna().mean():.3f} | late delivery (date rule) {x.late_delivery_date.mean():.3f}")
print("last delivered timestamp:", o.ts_delivered.max(), "| last carrier:", o.ts_carrier.max(), "| last review:", o.ts_review.max())
x = o[o.ts_purchase.dt.to_period("M") == "2018-08"]
print("Aug 2018 promised dates after data end (2018-10-17):", int((x.ts_estimated > "2018-10-17").sum()))

# E. Status is a final snapshot: does 'failed' leak into features?
print("\n=== E. Final status snapshot ===")
f = o[o.failed]
print(f"failed orders {len(f):,}: with handover {f.ts_carrier.notna().mean():.3f}, with approval {f.ts_approved.notna().mean():.3f}, "
      f"delivered {f.ts_delivered.notna().mean():.3f}")
print("orders with status shipped/invoiced/processing (stuck), by month:")
st = o[o.status.isin(["shipped", "invoiced", "processing", "created", "approved"])]
print(st.groupby([st.ts_purchase.dt.to_period("M"), "status"]).size().unstack(fill_value=0).tail(8).to_string())
print(f"shipped but never delivered, purchased before Jun 2018: {int(((o.status == 'shipped') & (o.ts_purchase < '2018-06-01')).sum()):,}")

# F. Multi review orders and the dedupe rule.
print("\n=== F. Reviews ===")
r = pd.read_csv(build.RAW / build.FILES["reviews"], parse_dates=["review_creation_date", "review_answer_timestamp"])
multi = r[r.order_id.duplicated(keep=False)]
g = multi.groupby("order_id").review_score.agg(["min", "max", "size"])
print(f"orders with several reviews {len(g):,}; differing scores {int((g['min'] != g['max']).sum()):,}")
dup = r[r.review_id.duplicated(keep=False)]
print(f"review_ids reused across orders: {dup.review_id.nunique():,} ids on {dup.order_id.nunique():,} orders "
      f"(same text and score copied to each order of a multi order purchase)")

# G. Customer repeat structure.
print("\n=== G. Customers ===")
c = pd.read_csv(build.RAW / build.FILES["customers"])
oc = o.merge(c[["customer_id", "customer_unique_id"]], left_on=o.order_id.map(
    pd.read_csv(build.RAW / build.FILES["orders"], usecols=["order_id", "customer_id"]).set_index("order_id").customer_id), right_on="customer_id", how="left")
n = oc.groupby("customer_unique_id").size()
print(f"people {len(n):,}; buying 2+ times {(n > 1).mean():.3%}; 3+ times {(n > 2).mean():.3%}")
