"""(a) What drives the order window models (permutation importance). (b) Holiday features: do they help?
(c) Natural shocks: Black Friday 2017 (demand) and the May 2018 truckers' strike (carrier)."""
import numpy as np, pandas as pd, holidays
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
exec(open("audit_05_orders_leads.py").read().split("tr_mask =")[0])   # reuse feature prep
br = holidays.Brazil(years=[2016,2017,2018])
hd = pd.to_datetime(list(br.keys()))
def days_to_next_hol(ts):
    idx = np.searchsorted(hd.values, ts.values); idx = np.clip(idx, 0, len(hd)-1)
    return (hd.values[idx] - ts.values)/np.timedelta64(1,"D")
b = b.sort_values("ts_purchase")
b["hol_in_dispatch"] = [sum(1 for h in hd if p.normalize() <= h <= l) if pd.notna(l) else 0 for p,l in zip(b.ts_purchase, b.ship_limit)]
b["days_to_hol"] = days_to_next_hol(b.ts_purchase)
b["bf_week"] = b.ts_purchase.between("2017-11-20","2017-11-30").astype(int)
tr = (b.ts_purchase>="2017-02-01")&(b.ts_purchase<"2018-01-01"); te=(b.ts_purchase>="2018-01-01")&(b.ts_purchase<"2018-07-01")
def fit(y, feats, imp=False):
    d=b[b[y].notna()]; a=d[tr[d.index]]; t=d[te[d.index]]
    m=HGBC(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=42, categorical_features="from_dtype").fit(a[feats],a[y])
    auc=roc_auc_score(t[y], m.predict_proba(t[feats])[:,1])
    if imp:
        s=t.sample(8000, random_state=1)
        pi=permutation_importance(m, s[feats], s[y], scoring="roc_auc", n_repeats=3, random_state=1)
        top=pd.Series(pi.importances_mean, feats).sort_values(ascending=False).head(8).round(4)
        return auc, top
    return auc
for y, extra, nm in [("late_ho",[],"DISPATCH window (at purchase)"),("late_del",["ho_delay_d"],"CARRIER window (at handover)")]:
    auc, top = fit(y, AT_PURCHASE+extra, imp=True)
    auc_h = fit(y, AT_PURCHASE+extra+["hol_in_dispatch","days_to_hol"])
    print(f"\n{nm}: AUC {auc:.3f} | with holiday features {auc_h:.3f}\n  top drivers (AUC lost when shuffled):\n{top.to_string()}")
# simple transparent baselines
d=b[b.late_ho.notna()&te]; print("\nDispatch: seller's own past late rate alone AUC", round(roc_auc_score(d.late_ho, d.s_late_ho_rate.fillna(b.late_ho.mean())),3))
d=b[b.late_del.notna()&te]; print("Carrier: promised days alone AUC", round(roc_auc_score(d.late_del, -d.promise_d),3),
      "| handover delay alone", round(roc_auc_score(d.late_del, d.ho_delay_d.fillna(0)),3))
# holiday descriptive
d=b[b.late_ho.notna()]
print("\nLate handover rate | holiday inside dispatch window:", d.groupby(d.hol_in_dispatch>0).late_ho.mean().round(3).to_dict(), "share of orders with one:", round((d.hol_in_dispatch>0).mean(),3))
# weekly shocks
b["wk"]=b.ts_purchase.dt.to_period("W")
w=b.groupby("wk").agg(orders=("order_id","size"), late_ho=("late_ho","mean"), late_del=("late_del","mean"), low_rev=("low_rev","mean"),
                       transit=("ts_delivered", lambda s: np.nan))
b["transit_d"]=(b.ts_delivered-b.ts_carrier).dt.total_seconds()/86400
w["transit_med"]=b.groupby("wk").transit_d.median()
print("\nWEEKLY around Black Friday 2017:"); print(w.loc["2017-10-30":"2017-12-31"].round(3).to_string())
print("\nWEEKLY around truckers' strike (21 May to 1 Jun 2018):"); print(w.loc["2018-04-23":"2018-07-01"].round(3).to_string())
print("\nBaseline 2018 H1 weekly medians: late_del", round(w.loc["2018-01-01":"2018-04-15"].late_del.median(),3), "low_rev", round(w.loc["2018-01-01":"2018-04-15"].low_rev.median(),3))
b.to_parquet("../data/interim/orders_features.parquet")
