"""(a) Shape of seller decline before loss: is there a window? (b) Doom loop: do ops/quality signals predict later decline?"""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).resolve().parent.parent / "data" / "interim"
b = pd.read_parquet(D/"orders_base.parquet"); b = b[b.seller_id.notna()]
b["m"] = pd.to_datetime(b.ts_purchase).dt.to_period("M")
months = pd.period_range("2016-09","2018-08",freq="M")
mc = b.groupby(["seller_id","m"]).size().unstack(fill_value=0).reindex(columns=months, fill_value=0)
A = mc.values
# loss event: last active month L followed by >=3 silent months observed (L <= 2018-05); seller had >=2 active months
rows=[]
for r, sid in enumerate(mc.index):
    act = np.nonzero(A[r])[0]
    if len(act) < 2: continue
    L = act[-1]
    if L+3 >= len(months): continue   # censored
    first = act[0]
    if L - first < 6: continue        # need 6 months history before loss
    traj = A[r, L-6:L+1]               # months L-6..L
    rows.append(dict(seller_id=sid, L=str(months[L]), peak=A[r,first:L+1].max(), life=A[r].sum(), **{f"m{k-6}":traj[k] for k in range(7)}))
T = pd.DataFrame(rows)
print("observed losses with >=6m history:", len(T))
big = T[T.life>=20]
print("of which life orders>=20:", len(big))
for nm, d in [("all", T), ("life>=20", big)]:
    cols=[f"m{k}" for k in range(-6,1)]
    norm = d[cols].div(d[cols].max(1).clip(lower=1), axis=0)
    print(nm, "mean orders by month before last active month (m0 = last month with an order):")
    print("  raw ", d[cols].mean().round(2).to_dict())
    print("  share of own peak", norm.mean().round(2).to_dict())
    prev3 = d[["m-5","m-4","m-3"]].sum(1); last3 = d[["m-2","m-1","m0"]].sum(1)
    print("  abrupt stop (last 3 months >= previous 3):", round((last3>=prev3).mean(),3),
          "| gradual decline (last3 < 0.6*prev3):", round((last3<0.6*prev3).mean(),3))
# doom loop: among steady sellers, does quality at T predict forward GMV ratio, controlling for volume band?
P = pd.read_parquet(D/"seller_panel.parquet")
P = P[P.g_fwd.notna() & (P.o_90>=10)].copy()
P["ratio"] = P.g_fwd/P.g_90
P["vol_band"] = pd.qcut(P.o_90, 4, labels=False, duplicates="drop")
for f in ["late_ho_rate","rev_low_share","rev_mean_shrunk","fail_rate","late_del_rate","overdue_now"]:
    x = P[P[f].notna()].copy()
    x["q"] = x.groupby("vol_band")[f].transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=False))
    g = x.groupby("q").agg(med_ratio=("ratio","median"), dormant=("dormant90","mean"), drop=("drop40","mean"), val=(f,"mean"))
    print(f"\n{f}: quartile (within volume band) -> median fwd/trailing GMV, dormancy, drop40  n={len(x)}")
    print(g.round(3).to_string())
