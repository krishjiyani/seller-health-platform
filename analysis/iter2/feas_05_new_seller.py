"""Feasibility F5: at day 30 after a seller's first sale, can we tell who will become an established seller?

Cohort: sellers whose first ever order (raw data, including 2016) falls in Mar 2017 to Feb 2018, so 180 days after it are observed.
Features use only what was known by day 30. Targets:
  established: 10 or more orders in days 31 to 180;  still_active: at least one order in days 91 to 180.
Baseline: orders in the first 30 days. Time split: first sale Mar to Oct 2017 (train) vs Nov 2017 to Feb 2018 (test).
Run: python -m analysis.iter2.feas_05_new_seller"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from analysis.iter2.base import load
from pipeline.build import RAW, FILES

o, _ = load()
items = pd.read_csv(RAW / FILES["items"], usecols=["order_id", "seller_id"])
raw_o = pd.read_csv(RAW / FILES["orders"], usecols=["order_id", "order_purchase_timestamp"], parse_dates=["order_purchase_timestamp"])
first = items.merge(raw_o, on="order_id").groupby("seller_id").order_purchase_timestamp.min().rename("first_sale")
deal_sellers = set(pd.read_csv(RAW / FILES["deals"], usecols=["seller_id"]).seller_id)
x = o[o.seller_id.notna()].merge(first, left_on="seller_id", right_index=True)
x["age"] = (x.ts_purchase - x.first_sale).dt.total_seconds() / 86400
coh = first[(first >= "2017-03-01") & (first < "2018-03-01")]
x = x[x.seller_id.isin(coh.index)]
D30 = pd.to_timedelta(30, unit="D")

w = x[x.age <= 30]
cut = w.first_sale + D30
g = w.groupby("seller_id")
f = pd.DataFrame({"o30": g.size(), "gmv30": g.price.sum(), "price_mean": g.price.mean(), "n_cat": g.category.nunique(),
                  "freight_ratio": (w.freight / w.price).groupby(w.seller_id).mean(), "same_state": g.same_state.mean(),
                  "dist": g.dist_km.mean(), "active_days": g.ts_purchase.apply(lambda s: s.dt.normalize().nunique())})
ho = w[w.ts_carrier <= cut]
f["late_disp"] = ho.groupby("seller_id").late_dispatch.mean()
f["handover_med"] = ho.groupby("seller_id").handover_d.median()
rv = w[w.ts_review <= cut]
f["n_rev"] = rv.groupby("seller_id").size()
f["rev_mean"] = rv.groupby("seller_id").review_score.mean()
f["low_share"] = rv.groupby("seller_id").low_review.mean()
second = x[x.age > 0].groupby("seller_id").age.min()
f["days_to_2nd"] = second.where(second <= 30)
f["first_month"] = coh.reindex(f.index).dt.month
f["deal_seller"] = f.index.isin(deal_sellers).astype(int)
st = x.groupby("seller_id").seller_state.first()
f["seller_state"] = st.reindex(f.index).astype("category")
f["main_cat"] = w.groupby("seller_id").category.agg(lambda s: s.mode().iloc[0]).astype(str).astype("category")
f = f.reindex(coh.index)            # sellers whose first order was attributed to a co seller have no row: keep them with o30 = 0
f["o30"] = f.o30.fillna(0)
f["n_rev"] = f.n_rev.fillna(0)
later = x[(x.age > 30) & (x.age <= 180)].groupby("seller_id").size().reindex(coh.index).fillna(0)
act = x[(x.age > 90) & (x.age <= 180)].groupby("seller_id").size().reindex(coh.index).fillna(0)
f["established"] = (later >= 10).astype(int)
f["still_active"] = (act > 0).astype(int)
f["first_sale"] = coh
print(f"cohort sellers {len(f):,}; established {f.established.mean():.3f}; still active days 91 to 180 {f.still_active.mean():.3f}")
print("established rate by orders in first 30 days:", f.groupby(f.o30.clip(upper=6)).established.agg(["mean", "size"]).round(3).to_dict("index"))

F = ["o30", "gmv30", "price_mean", "n_cat", "freight_ratio", "same_state", "dist", "active_days", "late_disp", "handover_med",
     "n_rev", "rev_mean", "low_share", "days_to_2nd", "first_month", "deal_seller", "seller_state", "main_cat"]
tr, te = f[f.first_sale < "2017-11-01"], f[f.first_sale >= "2017-11-01"]
print(f"train {len(tr):,} | test {len(te):,}")
for y in ["established", "still_active"]:
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30, random_state=42,
                                         categorical_features="from_dtype").fit(tr[F], tr[y])
    p = clf.predict_proba(te[F])[:, 1]
    top = p >= np.quantile(p, 0.8)
    print(f"{y}: test rate {te[y].mean():.3f} | model AUC {roc_auc_score(te[y], p):.3f} | baseline (orders in 30 days) "
          f"{roc_auc_score(te[y], te.o30):.3f} | baseline (GMV 30 days) {roc_auc_score(te[y], te.gmv30.fillna(0)):.3f} | "
          f"top 20% rate {te[y].values[top].mean():.3f}")
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(clf, te[F], te[y], scoring="roc_auc", n_repeats=5, random_state=42)
    imp = pd.Series(pi.importances_mean, index=F).sort_values(ascending=False).head(6)
    print("   top drivers:", imp.round(3).to_dict())
