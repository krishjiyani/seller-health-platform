"""Order-level and lead-level predictability. Time split: train purchases 2017, test Jan-Jun 2018.
Seller history features come from the seller panel at the previous month end (no leakage)."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from sklearn.metrics import roc_auc_score, average_precision_score
D = Path(__file__).resolve().parent.parent / "data" / "interim"
b = pd.read_parquet(D/"orders_base.parquet")
for c in ["ts_purchase","ts_carrier","ts_delivered","ts_estimated","ship_limit","ts_approved"]: b[c]=pd.to_datetime(b[c])
b = b[b.seller_id.notna()].copy()
b["m"] = b.ts_purchase.dt.to_period("M"); b["prevM"] = b.m - 1
P = pd.read_parquet(D/"seller_panel.parquet")
P = P[["seller_id","M","o_90","late_ho_rate","rev_mean_shrunk","fail_rate","tenure_d","late_del_rate","life_orders"]]
P.columns = ["seller_id","prevM"]+["s_"+c for c in P.columns[2:]]
b = b.merge(P, on=["seller_id","prevM"], how="left")
b["new_seller"] = b.s_o_90.isna().astype(int)
b["promise_d"] = (b.ts_estimated - b.ts_purchase).dt.days
b["limit_d"] = (b.ship_limit - b.ts_purchase).dt.days
b["dow"] = b.ts_purchase.dt.dayofweek; b["hour"]=b.ts_purchase.dt.hour; b["moy"]=b.ts_purchase.dt.month
b["same_state"] = (b.customer_state==b.seller_state).astype(int)
for c in ["category","customer_state","seller_state","pay_type"]: b[c]=b[c].astype("category")
b["late_ho"] = (b.ts_carrier > b.ship_limit).astype(float).where(b.ts_carrier.notna())
b["late_del"] = (b.ts_delivered > b.ts_estimated).astype(float).where(b.ts_delivered.notna())
b["low_rev"] = (b.review_score<=2).astype(float).where(b.review_score.notna())
b["fail"] = b.status.isin(["canceled","unavailable"]).astype(float)
b["ho_delay_d"] = (b.ts_carrier - b.ship_limit).dt.total_seconds()/86400
b["deliv_delay_d"] = (b.ts_delivered - b.ts_estimated).dt.total_seconds()/86400
AT_PURCHASE = ["price","freight","n_items","n_sellers","weight_g","photos","desc_len","dist_km","same_state","promise_d","limit_d",
               "dow","hour","moy","installments","pay_value","new_seller"]+[c for c in b.columns if c.startswith("s_")]+["category","customer_state","seller_state","pay_type"]
tr_mask = (b.ts_purchase>="2017-02-01")&(b.ts_purchase<"2018-01-01"); te_mask=(b.ts_purchase>="2018-01-01")&(b.ts_purchase<"2018-07-01")
out=[]
def test(name, y, feats, cond=None):
    d = b if cond is None else b[cond]
    tr = d[tr_mask[d.index] & d[y].notna()]; te = d[te_mask[d.index] & d[y].notna()]
    m = HGBC(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=42, categorical_features="from_dtype").fit(tr[feats], tr[y])
    p = m.predict_proba(te[feats])[:,1]
    top = te.assign(p=p).nlargest(len(te)//10,"p")[y].mean()
    r = dict(target=name, n_test=len(te), base=round(te[y].mean(),3), auc=round(roc_auc_score(te[y],p),3),
             pr_auc=round(average_precision_score(te[y],p),3), top10_rate=round(top,3)); out.append(r); print(r)
test("O1 late seller handover, at purchase", "late_ho", AT_PURCHASE)
test("O2 late customer delivery, at purchase", "late_del", AT_PURCHASE)
test("O3 late delivery, at carrier handover (knows handover delay)", "late_del", AT_PURCHASE+["ho_delay_d"])
test("O4 1-2 star review, at purchase", "low_rev", AT_PURCHASE)
test("O5 1-2 star review, at delivery (knows delays)", "low_rev", AT_PURCHASE+["ho_delay_d","deliv_delay_d"])
test("O6 cancel/unavailable, at purchase", "fail", AT_PURCHASE)
# repeat customer
c = b.sort_values("ts_purchase").groupby("customer_unique_id")
b["next_buy"] = c.ts_purchase.shift(-1); b["repeat180"] = ((b.next_buy - b.ts_purchase).dt.days <= 180).astype(float)
b.loc[b.ts_purchase >= "2018-03-01","repeat180"] = np.nan
first = b.groupby("customer_unique_id").cumcount()==0
tm = te_mask.copy(); te_mask = (b.ts_purchase>="2018-01-01")&(b.ts_purchase<"2018-03-01")
test("O7 customer buys again within 180d (first orders)", "repeat180", AT_PURCHASE+["review_score"], cond=first)
te_mask = tm
# low review explained: how much of 1-2 star is delivery?
dd = b[b.low_rev.notna() & b.late_del.notna()]
print("\n1-2 star rate | on time:", round(dd[dd.late_del==0].low_rev.mean(),3), "| late:", round(dd[dd.late_del==1].low_rev.mean(),3))
print("share of all 1-2 star reviews that are on late orders:", round(dd[dd.low_rev==1].late_del.mean(),3))
print("1-2 star on late orders where seller handover was ON TIME (unjust):", int(((dd.low_rev==1)&(dd.late_del==1)&(dd.late_ho==0)).sum()))
print("non delivered orders 1-2 star rate:", round(b[b.ts_delivered.isna() & b.review_score.notna()].low_rev.mean(),3))
# leads
from sklearn.model_selection import StratifiedKFold, cross_val_score
mql = pd.read_parquet(D/"mql.parquet"); deals = pd.read_parquet(D/"deals.parquet")
mql["won"] = mql.mql_id.isin(deals.mql_id).astype(int); mql["fc"]=pd.to_datetime(mql.first_contact_date)
mql["moy"]=mql.fc.dt.month; mql["dow"]=mql.fc.dt.dayofweek
lp = mql.landing_page_id.value_counts(); mql["lp"] = mql.landing_page_id.where(mql.landing_page_id.map(lp)>=30,"other")
X = pd.get_dummies(mql[["origin","lp"]].fillna("na")).join(mql[["moy","dow"]])
auc = cross_val_score(HGBC(max_iter=200, random_state=42), X, mql.won, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc")
print(f"\nL1 MQL -> won (origin, landing page, timing) 5 fold AUC {auc.mean():.3f} +- {auc.std():.3f}, base {mql.won.mean():.3f}")
# activation: won -> first sale within 90d (runway adjusted)
first_sale = b.groupby("seller_id").ts_purchase.min()
deals["won"] = pd.to_datetime(deals.won_date); deals["fs"] = deals.seller_id.map(first_sale)
el = deals[deals.won <= "2018-06-01"].copy(); el["act90"] = ((el.fs - el.won).dt.days.between(0,90)).astype(int)
cats = ["lead_type","business_type","business_segment","origin","lead_behaviour_profile"]
Xd = pd.get_dummies(el[cats].fillna("na"))
from sklearn.linear_model import LogisticRegression
auc = cross_val_score(LogisticRegression(C=0.5, max_iter=2000), Xd, el.act90, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc")
print(f"L2 won -> first sale within 90d (n={len(el)}, base {el.act90.mean():.3f}) logistic AUC {auc.mean():.3f} +- {auc.std():.3f}")
lag = (el.fs-el.won).dt.days.dropna(); print("activation lag days: median", lag.median(), "p75", lag.quantile(.75), "p90", lag.quantile(.9))
# survival of activation: if not sold by day d, P(sell by day 90)?
for d in [15,30,44,60]:
    alive = el[~((el.fs-el.won).dt.days < d)]
    print(f"  not sold by day {d}: n={len(alive)}, later sold within 90d = {((alive.fs-alive.won).dt.days.between(d,90)).mean():.3f}")
pd.DataFrame(out).to_csv(D/"audit_order_summary.csv", index=False)
