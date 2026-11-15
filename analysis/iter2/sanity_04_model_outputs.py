"""Data sanity v2, part 4: audit the iteration 1 scores the app shows (data/app), without retraining.

* Calibration: when the app says 40%, does it happen 40% of the time?
* Do the AUCs hold under corrected labels (date rule, lost parcels counted, clean August cohort)?
Run: python -m analysis.iter2.sanity_04_model_outputs"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from pipeline.config import APP

o = pd.read_parquet(APP / "orders.parquet")
CUT = pd.Timestamp("2018-10-17")
o["late_date"] = (o.ts_delivered.dt.normalize() > o.ts_estimated).astype(float).where(o.ts_delivered.notna())
lost = o.ts_carrier.notna() & o.ts_delivered.isna() & ~o.failed
o["late_or_lost"] = o.late_date.where(~lost, 1.0)
never = o.ts_carrier.isna() & ~o.failed & o.ship_limit.notna()
o["late_or_never_dispatch"] = o.late_dispatch.where(~never, 1.0)
o["clean_cohort"] = o.ts_estimated <= "2018-08-29"


def calib(y, p, bins=(0, .05, .1, .2, .3, .5, .7, 1.0)):
    b = pd.cut(p, bins, include_lowest=True)
    t = pd.DataFrame({"y": y, "p": p, "b": b}).groupby("b", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    return t.round(3)


for window, label_old, labels_new in [("dispatch", "late_dispatch", ["late_or_never_dispatch"]),
                                      ("carrier", "late_delivery", ["late_date", "late_or_lost"])]:
    d = pd.read_parquet(APP / f"daily_{window}.parquet")
    d["day"] = pd.to_datetime(d.day)
    d = d[d.day >= "2017-09-01"].merge(o[["order_id", label_old, "clean_cohort"] + labels_new], on="order_id", how="left")
    print(f"\n=== {window.upper()} daily scores (Sep 2017 on), {len(d):,} order days ===")
    k = d[d[label_old].notna()]
    print(f"old label {label_old}: pooled AUC {roc_auc_score(k[label_old], k.p):.3f} | Brier {brier_score_loss(k[label_old], k.p):.4f} | "
          f"mean p {k.p.mean():.3f} vs rate {k[label_old].mean():.3f}")
    print(calib(k[label_old], k.p).to_string())
    for lab in labels_new:
        k2 = d[d[lab].notna()]
        print(f"new label {lab}: pooled AUC {roc_auc_score(k2[lab], k2.p):.3f} | rate {k2[lab].mean():.3f} | n {len(k2):,}")
        k3 = k2[k2.clean_cohort]
        print(f"   clean cohort only: AUC {roc_auc_score(k3[lab], k3.p):.3f} | rate {k3[lab].mean():.3f}")
    by_m = k.groupby(k.day.dt.to_period("M")).apply(lambda g: pd.Series({"n": len(g), "rate": g[label_old].mean(), "mean_p": g.p.mean(),
                                                                       "auc": roc_auc_score(g[label_old], g.p)}), include_groups=False)
    print("by month (old label):\n" + by_m.round(3).to_string())

sm = pd.read_parquet(APP / "seller_months.parquet")
print("\n=== SILENCE scores ===")
for k in (1, 3):
    s = sm[sm[f"p_silent_{k}"].notna() & sm[f"silent_by_{k}"].notna()]
    print(f"horizon {k}: n {len(s):,} | AUC {roc_auc_score(s[f'silent_by_{k}'], s[f'p_silent_{k}']):.3f} | "
          f"mean p {s[f'p_silent_{k}'].mean():.3f} vs rate {s[f'silent_by_{k}'].mean():.3f}")
    print(calib(s[f"silent_by_{k}"], s[f"p_silent_{k}"]).to_string())
s = sm[sm.p_silent_1.notna()]
print("sellers scored per month:", s.groupby("month").size().to_dict())
big = s[s.o_90 >= 10]
kk = big[big.silent_by_3.notna()]
print(f"sellers with 10+ orders in 90 days: AUC silent_by_3 {roc_auc_score(kk.silent_by_3, kk.p_silent_3):.3f} "
      f"recency AUC {roc_auc_score(kk.silent_by_3, kk.recency_d):.3f}, rate {kk.silent_by_3.mean():.3f}, n {len(kk):,}")
small = s[(s.o_90 < 3)]
kk = small[small.silent_by_3.notna()]
print(f"sellers with 1 to 2 orders in 90 days: rate silent_by_3 {kk.silent_by_3.mean():.3f}, n {len(kk):,}, AUC {roc_auc_score(kk.silent_by_3, kk.p_silent_3):.3f}")
gmv = s[s.silent_by_3.notna()]
print(f"share of 90 day GMV held by sellers that went silent within 3 months: {(gmv.g_90 * gmv.silent_by_3).sum() / gmv.g_90.sum():.3f}")
