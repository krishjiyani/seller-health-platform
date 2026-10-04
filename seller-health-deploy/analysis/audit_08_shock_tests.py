"""Two candidate hero findings from natural shocks.
H1 Black Friday stress test: did sellers whose dispatch broke during the BF surge (20 Nov to 3 Dec 2017) decline afterwards,
   compared with sellers of similar size whose dispatch held? Does BF lateness add over normal lateness?
H2 Truckers' strike: sellers with more customer-late, seller-on-time orders around the strike: did their reviews and later GMV suffer?"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
b = pd.read_parquet("../data/interim/orders_features.parquet")
g = lambda d: d.groupby("seller_id")
def window(a, z): return b[(b.ts_purchase>=a)&(b.ts_purchase<z)]
pre = window("2017-09-01","2017-11-20"); bf = window("2017-11-20","2017-12-04"); post = window("2018-01-01","2018-04-01")
S = pd.DataFrame({"pre_orders": g(pre).size(), "pre_late": g(pre).late_ho.mean(), "pre_gmv": g(pre).price.sum(),
                  "bf_orders": g(bf).size(), "bf_late": g(bf).late_ho.mean(), "post_gmv": g(post).price.sum(), "post_orders": g(post).size()}).fillna({"post_gmv":0,"post_orders":0})
S = S[(S.pre_orders>=10)&(S.bf_orders>=5)].copy()
S["growth"] = (S.post_gmv/ (S.pre_gmv*90/80))       # per day normalised (80 day pre window, 90 day post)
S["dormant"] = (S.post_orders==0).astype(int); S["drop"] = (S.growth<0.6).astype(int)
S["broke"] = ((S.bf_late - S.pre_late.fillna(0)) >= 0.15).astype(int)
print("H1 sellers:", len(S), "| broke under BF (late rate +15pts):", S.broke.sum())
print(S.groupby("broke").agg(n=("growth","size"), med_growth=("growth","median"), drop_rate=("drop","mean"), dormant=("dormant","mean"),
                             pre_late=("pre_late","mean"), bf_late=("bf_late","mean")).round(3).to_string())
S["vol_band"]=pd.qcut(S.pre_orders,3,labels=False)
print("within volume band (median growth broke vs held):", S.groupby(["vol_band","broke"]).growth.median().unstack().round(2).to_dict())
print("AUC for drop: pre late rate", round(roc_auc_score(S["drop"], S.pre_late.fillna(0)),3), "| BF late rate", round(roc_auc_score(S["drop"], S.bf_late.fillna(0)),3),
      "| BF minus pre", round(roc_auc_score(S["drop"], (S.bf_late-S.pre_late.fillna(0))),3))
# H2 strike
st = window("2018-05-07","2018-05-28")
st = st[st.late_ho==0]   # seller dispatched on time
E = pd.DataFrame({"st_n": g(st).size(), "st_carrier_late": g(st).late_del.mean(), "st_lowrev": g(st).low_rev.mean()})
pre2 = window("2018-02-01","2018-05-01"); post2 = window("2018-06-01","2018-08-01")
E["pre_gmv"]=g(pre2).price.sum(); E["post_gmv"]=g(post2).price.sum(); E=E.fillna({"post_gmv":0})
E = E[(E.st_n>=5)&(E.pre_gmv>0)].copy()
E["growth"]=(E.post_gmv/61)/(E.pre_gmv/89)
E["hit"]=(E.st_carrier_late>=0.2).astype(int)
print("\nH2 sellers with >=5 on-time-dispatched orders during the strike:", len(E), "| hit (>=20% of them arrived late anyway):", E.hit.sum())
print(E.groupby("hit").agg(n=("growth","size"), carrier_late=("st_carrier_late","mean"), low_review=("st_lowrev","mean"), med_growth=("growth","median")).round(3).to_string())
