"""Feasibility F7: do bad reviews cost a seller future sales? A natural experiment broader than the truckers' strike.

Design: seller weeks in which the seller had at least one CARRIER caused late delivery reviewed that week (seller dispatched on
time, parcel arrived after the promise date). Treated: at least one of those got 1 or 2 stars. Control: all of them got 3 to 5.
Lateness is the same kind in both groups and not the seller's doing, so the difference is mainly the customer's reaction.
Outcome: log of (orders in the 4 weeks after + 1) / (orders in the 4 weeks before + 1). Sellers with 8+ orders in the 4 weeks before.
Run: python -m analysis.iter2.feas_07_review_effect"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from analysis.iter2.base import load

o, _ = load()
o = o[o.seller_id.notna() & o.clean].copy()
o["wk"] = o.ts_purchase.dt.to_period("W-SUN")
cnt = o.groupby(["seller_id", "wk"]).size().unstack(fill_value=0)
weeks = pd.period_range("2017-01-02", "2018-08-26", freq="W-SUN")
cnt = cnt.reindex(columns=weeks, fill_value=0)
cf = o[(o.late_dispatch == 0) & (o.late_date == 1) & o.ts_review.notna()].copy()
cf["rwk"] = cf.ts_review.dt.to_period("W-SUN")
g = cf.groupby(["seller_id", "rwk"]).agg(n_cf=("order_id", "size"), n_low=("low_review", "sum")).reset_index()
pos = {w: i for i, w in enumerate(weeks)}
rows = []
A = cnt.values
sidx = {s: i for i, s in enumerate(cnt.index)}
for r in g.itertuples():
    if r.rwk not in pos or r.seller_id not in sidx:
        continue
    j, i = pos[r.rwk], sidx[r.seller_id]
    if j < 4 or j + 4 >= len(weeks) or weeks[j + 4].end_time > pd.Timestamp("2018-07-31"):
        continue
    before, after = A[i, j - 4:j].sum(), A[i, j + 1:j + 5].sum()
    if before < 8:
        continue
    rows.append(dict(seller_id=r.seller_id, wk=str(r.rwk), treated=int(r.n_low > 0), n_cf=r.n_cf, before=before, after=after,
                     y=np.log((after + 1) / (before + 1))))
d = pd.DataFrame(rows)
print(f"seller weeks {len(d):,} ({d.seller_id.nunique():,} sellers); treated {d.treated.sum():,}, control {(1 - d.treated).sum():,}")
print(d.groupby("treated")[["before", "after", "n_cf", "y"]].mean().round(3).to_string())
rng = np.random.default_rng(0)
diffs = []
t, c = d[d.treated == 1].y.values, d[d.treated == 0].y.values
for _ in range(2000):
    diffs.append(rng.choice(t, len(t)).mean() - rng.choice(c, len(c)).mean())
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"difference in log growth (treated minus control): {t.mean() - c.mean():.3f}, 95% CI [{lo:.3f}, {hi:.3f}] "
      f"(about {np.exp(t.mean() - c.mean()) - 1:+.1%} orders)")
# adjust for prior volume and number of carrier late orders with a simple regression
import numpy.linalg as la
X = np.column_stack([np.ones(len(d)), d.treated, np.log(d.before), np.log(d.n_cf)])
beta, *_ = la.lstsq(X, d.y.values, rcond=None)
res = d.y.values - X @ beta
cov = la.inv(X.T @ X) * (res @ res) / (len(d) - X.shape[1])
print(f"adjusted effect of a bad review: {beta[1]:.3f} (se {np.sqrt(cov[1, 1]):.3f})")
big = d[d.before >= 20]
if len(big) > 50:
    print(f"sellers with 20+ prior orders: n {len(big)}, diff {big[big.treated == 1].y.mean() - big[big.treated == 0].y.mean():.3f}")
