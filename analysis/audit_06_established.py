"""Established sellers (>=10 orders in trailing 90d): can we predict decline (GMV drop >40% or dormant), and do ops signals add over activity?
Also lead time: for established sellers later lost, how many months ahead did the model rank them in the top 20%?"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from sklearn.metrics import roc_auc_score
D = Path(__file__).resolve().parent.parent / "data" / "interim"
P = pd.read_parquet(D/"seller_panel.parquet")
FEAT = ["tenure_d","recency_d","life_orders","o_0_30","o_30_60","o_60_90","o_prev90","g_90","g_prev90","trend","mom",
        "active_m6","peak_m_orders","last_m_vs_peak","ho_n","late_ho_rate","overdue_now","late_del_rate","carrier_blame",
        "rev_n","rev_mean_shrunk","rev_low_share","fail_rate","avg_price","freight_ratio","n_cat","dist_km","moy"]
OPS = ["late_ho_rate","overdue_now","late_del_rate","carrier_blame","rev_n","rev_mean_shrunk","rev_low_share","fail_rate"]
PRM = dict(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=40, random_state=42)
E = P[P.o_90>=10].copy()
def wf(feats, y="lost90"):
    a=[]; preds=[]
    for M in pd.period_range("2017-09","2018-05",freq="M"):
        tr=E[(E.M<=M-3)&E[y].notna()]; te=E[(E.M==M)&E[y].notna()]
        p=HGBC(**PRM).fit(tr[feats],tr[y]).predict_proba(te[feats])[:,1]
        a.append(roc_auc_score(te[y],p)); preds.append(te.assign(p=p))
    return np.mean(a), np.std(a), pd.concat(preds)
full = wf(FEAT); act = wf([f for f in FEAT if f not in OPS])
print(f"established n/fold ~{int(len(E[E.M=='2018-01']))}, lost90 base {E.lost90.mean():.3f}")
print(f"lost90 AUC full {full[0]:.3f}+-{full[1]:.3f} | activity only {act[0]:.3f} | recency only", end=" ")
pr = full[2]; print(round(np.mean([roc_auc_score(g.lost90, g.recency_d) for _,g in pr.groupby('M')]),3))
pr["rk"] = pr.groupby("M").p.rank(pct=True)
top = pr[pr.rk>=0.8]; bot = pr[pr.rk<=0.2]
print(f"top 20% risk: lost rate {top.lost90.mean():.3f}, GMV retained median {(top.g_fwd/top.g_90).median():.2f} | bottom 20%: lost {bot.lost90.mean():.3f}, retained {(bot.g_fwd/bot.g_90).median():.2f}")
print(f"lift top20 vs bottom20: {top.lost90.mean()/bot.lost90.mean():.1f}x")
print("GMV at stake in top 20% per month (median R$):", round(top.groupby('M').g_90.sum().median()))
