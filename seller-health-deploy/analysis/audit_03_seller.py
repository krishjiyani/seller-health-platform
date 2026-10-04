"""Seller-side predictability audit. Panel of (seller, origin month end T).
Features use only info timestamped <= T. Labels from months after T.
Walk-forward: test origin M trained on origins whose label window closed by M."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, HistGradientBoostingRegressor as HGBR
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from scipy.stats import spearmanr
D = Path(__file__).resolve().parent.parent / "data" / "interim"
b = pd.read_parquet(D / "orders_base.parquet")
b = b[b.seller_id.notna()].copy()
for c in ["ts_purchase","ts_carrier","ts_delivered","ts_estimated","ship_limit","review_creation_date","ts_approved"]:
    b[c] = pd.to_datetime(b[c])
b["m"] = b.ts_purchase.dt.to_period("M")
b["late_ho"] = b.ts_carrier > b.ship_limit
b["late_del"] = b.ts_delivered > b.ts_estimated
b["fail"] = b.status.isin(["canceled","unavailable"])
# monthly order counts per seller for labels
months = pd.period_range("2016-09","2018-08",freq="M")
mc = b.groupby(["seller_id","m"]).size().unstack(fill_value=0).reindex(columns=months, fill_value=0)
mg = b.groupby(["seller_id","m"]).price.sum().unstack(fill_value=0).reindex(columns=months, fill_value=0)
MI = {p:i for i,p in enumerate(months)}
arr = mc.values; garr = mg.values
PRIOR = b.review_score.mean(); C = 10

def feats(M):
    T = M.to_timestamp(how="end")
    h = b[b.ts_purchase <= T]
    w90 = h[h.ts_purchase > T - pd.Timedelta(days=90)]
    w180 = h[h.ts_purchase > T - pd.Timedelta(days=180)]
    g = h.groupby("seller_id")
    f = pd.DataFrame({"first": g.ts_purchase.min(), "last": g.ts_purchase.max(), "life_orders": g.size()})
    f["tenure_d"] = (T - f["first"]).dt.days; f["recency_d"] = (T - f["last"]).dt.days
    for lo, hi, nm in [(0,30,"o_0_30"),(30,60,"o_30_60"),(60,90,"o_60_90"),(90,180,"o_prev90")]:
        x = h[(h.ts_purchase > T - pd.Timedelta(days=hi)) & (h.ts_purchase <= T - pd.Timedelta(days=lo))]
        f[nm] = x.groupby("seller_id").size(); f["g"+nm[1:]] = x.groupby("seller_id").price.sum()
    f = f.fillna({c:0 for c in f.columns if c.startswith(("o_","g_"))})
    f["o_90"] = f.o_0_30+f.o_30_60+f.o_60_90; f["g_90"] = f.g_0_30+f.g_30_60+f.g_60_90
    f["trend"] = (f.g_90 - f.g_prev90)/(f.g_prev90 + f.g_90 + 1)
    f["mom"] = (f.o_0_30 - f.o_60_90)/(f.o_90+1)
    i = MI[M]; f["active_m6"] = pd.Series((arr[:, max(0,i-5):i+1] > 0).sum(1), index=mc.index)
    f["peak_m_orders"] = pd.Series(arr[:, :i+1].max(1), index=mc.index)
    f["last_m_vs_peak"] = pd.Series(arr[:, i], index=mc.index)/f.peak_m_orders.clip(lower=1)
    # fulfilment: handovers known by T, purchased in last 180d
    ho = w180[w180.ts_carrier <= T]; f["ho_n"] = ho.groupby("seller_id").size()
    f["late_ho_rate"] = ho.groupby("seller_id").late_ho.mean()
    od = w90[(w90.ship_limit < T) & ((w90.ts_carrier.isna()) | (w90.ts_carrier > T))]
    f["overdue_now"] = od.groupby("seller_id").size()
    dl = w180[w180.ts_delivered <= T]; f["late_del_rate"] = dl.groupby("seller_id").late_del.mean()
    f["carrier_blame"] = dl[dl.late_del].assign(x=lambda d: ~d.late_ho).groupby("seller_id").x.mean()
    rv = w180[w180.review_creation_date <= T]; rg = rv.groupby("seller_id").review_score
    f["rev_n"] = rg.count(); f["rev_mean_shrunk"] = (C*PRIOR + rg.sum())/(C + f.rev_n.fillna(0))
    f["rev_low_share"] = rv.assign(x=rv.review_score<=2).groupby("seller_id").x.mean()
    cx = w90[w90.ts_purchase <= T - pd.Timedelta(days=7)]
    f["fail_rate"] = cx.groupby("seller_id").fail.mean()
    f["avg_price"] = w90.groupby("seller_id").price.mean()
    f["freight_ratio"] = (w90.freight/w90.price).groupby(w90.seller_id).mean()
    f["n_cat"] = w90.groupby("seller_id").category.nunique()
    f["dist_km"] = w90.groupby("seller_id").dist_km.mean()
    f["moy"] = M.month
    f = f[f.o_90 > 0].copy()
    f["rev_mean_shrunk"] = f.rev_mean_shrunk.fillna(PRIOR)
    # labels
    rows = mc.index.get_indexer(f.index)
    def silent_from(s):  # months s..s+2 all zero, needs s+2 observed
        if s+2 >= len(months): return np.full(len(rows), np.nan)
        return (arr[rows, s:s+3].sum(1) == 0).astype(float)
    f["dormant90"] = silent_from(i+1)
    fwd_g = garr[rows, i+1:i+4].sum(1) if i+3 < len(months) else np.full(len(rows), np.nan)
    f["g_fwd"] = fwd_g
    f["drop40"] = np.where(np.isnan(fwd_g), np.nan, (fwd_g < 0.6*f.g_90.values).astype(float))
    f["lost90"] = np.where(np.isnan(fwd_g), np.nan, np.maximum(f.dormant90, f.drop40))
    # survival: loss month s = first month > M starting a 3 month silence
    lb = {}
    for k in (1,2,3):
        cols = [silent_from(i+j) for j in range(1,k+1)]
        lb[k] = np.nanmax(np.vstack(cols),0) if not any(np.isnan(c).all() for c in cols) else np.full(len(rows), np.nan)
        f[f"lost_by_{k}"] = lb[k]
    f["M"] = M
    return f.reset_index(names="seller_id")

origins = pd.period_range("2017-03","2018-07",freq="M")
P = pd.concat([feats(M) for M in origins], ignore_index=True)
P.to_parquet(D/"seller_panel.parquet")
print("panel rows", len(P), "sellers", P.seller_id.nunique())
FEAT = ["tenure_d","recency_d","life_orders","o_0_30","o_30_60","o_60_90","o_prev90","g_90","g_prev90","trend","mom",
        "active_m6","peak_m_orders","last_m_vs_peak","ho_n","late_ho_rate","overdue_now","late_del_rate","carrier_blame",
        "rev_n","rev_mean_shrunk","rev_low_share","fail_rate","avg_price","freight_ratio","n_cat","dist_km","moy"]
PRM = dict(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=40, random_state=42)

def walk(target, subset=None, test_from="2017-09", test_to="2018-05", horizon=3, feats=FEAT):
    res = []; allp = []
    for M in pd.period_range(test_from, test_to, freq="M"):
        tr = P[(P.M <= M - horizon) & P[target].notna()]; te = P[(P.M == M) & P[target].notna()]
        if subset is not None: tr, te = tr[subset(tr)], te[subset(te)]
        if te[target].nunique() < 2 or len(tr) < 200: continue
        mdl = HGBC(**PRM).fit(tr[feats], tr[target])
        p = mdl.predict_proba(te[feats])[:,1]
        res.append(dict(M=str(M), n=len(te), base=te[target].mean(), auc=roc_auc_score(te[target], p),
                        auc_recency=roc_auc_score(te[target], te.recency_d),
                        ap=average_precision_score(te[target], p)))
        allp.append(te.assign(p=p))
    r = pd.DataFrame(res)
    return r, pd.concat(allp)

summary = []
def report(name, target, **kw):
    r, ap = walk(target, **kw)
    top = ap.groupby("M", group_keys=False).apply(lambda d: d.nlargest(max(1,len(d)//5), "p"))
    bot = ap.groupby("M", group_keys=False).apply(lambda d: d.nsmallest(max(1,len(d)//5), "p"))
    s = dict(target=name, folds=len(r), n_test=int(r.n.sum()), base=round(ap[target].mean(),3),
             auc_mean=round(r.auc.mean(),3), auc_sd=round(r.auc.std(),3), auc_recency_only=round(r.auc_recency.mean(),3),
             pr_auc=round(r.ap.mean(),3), top20_rate=round(top[target].mean(),3), bottom20_rate=round(bot[target].mean(),3))
    summary.append(s); print(s); return ap

ap_lost = report("lost90 (dormant OR GMV drop >40%)", "lost90")
report("dormant90", "dormant90")
report("drop40 among non dormant", "drop40", subset=lambda d: d.dormant90==0)
report("lost_by_1 (silence starts M+1)", "lost_by_1")
report("lost_by_2", "lost_by_2", test_to="2018-04", horizon=4)
report("lost_by_3", "lost_by_3", test_to="2018-03", horizon=5)
report("dormant90, sellers with >=10 lifetime orders", "dormant90", subset=lambda d: d.life_orders>=10)
report("dormant90, sellers with >=30 orders in 90d (big)", "dormant90", subset=lambda d: d.o_90>=30)
report("dormant90, new sellers tenure<=60d", "dormant90", subset=lambda d: d.tenure_d<=60)
# ablation: what drives it
OPS = ["late_ho_rate","overdue_now","late_del_rate","carrier_blame","rev_n","rev_mean_shrunk","rev_low_share","fail_rate"]
r_noops,_ = walk("lost90", feats=[f for f in FEAT if f not in OPS])
r_ops,_ = walk("lost90", feats=OPS+["moy"])
print("ABLATION lost90: activity only AUC", round(r_noops.auc.mean(),3), "| ops/quality only AUC", round(r_ops.auc.mean(),3))
r_noops,_ = walk("dormant90", subset=lambda d: d.life_orders>=10, feats=[f for f in FEAT if f not in OPS])
print("ABLATION dormant90 >=10 orders: activity only AUC", round(r_noops.auc.mean(),3))
# GMV regression
sp=[]
for M in pd.period_range("2017-09","2018-05",freq="M"):
    tr = P[(P.M <= M-3) & P.g_fwd.notna()]; te = P[(P.M==M)&P.g_fwd.notna()]
    mdl = HGBR(**PRM).fit(tr[FEAT], np.log1p(tr.g_fwd)); p = mdl.predict(te[FEAT])
    sp.append((spearmanr(p, te.g_fwd).correlation, spearmanr(te.g_90, te.g_fwd).correlation))
sp = np.array(sp); print("GMV next 90d spearman model", sp[:,0].mean().round(3), "naive (last 90d gmv)", sp[:,1].mean().round(3))
pd.DataFrame(summary).to_csv(D/"audit_seller_summary.csv", index=False)
