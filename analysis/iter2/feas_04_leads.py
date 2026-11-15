"""Feasibility F4: the pre signing funnel (MQL to won deal) and the sales rep effect on activation.

Q1. Is the MQL win rate observable for every MQL month, or truncated by the deals file starting in Dec 2017?
Q2. Lead decay: P(won eventually | not won by day d).
Q3. Lead scoring at first contact: AUC on a time split, by feature group.
Q4. Do sales reps differ in how often their signed sellers activate (first sale within 90 days)?
Run: python -m analysis.iter2.feas_04_leads"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from pipeline.build import RAW, FILES

mql = pd.read_csv(RAW / FILES["mql"], parse_dates=["first_contact_date"])
deals = pd.read_csv(RAW / FILES["deals"], parse_dates=["won_date"])
m = mql.merge(deals[["mql_id", "won_date", "sr_id", "sdr_id"]], on="mql_id", how="left")
m["won"] = m.won_date.notna().astype(int)
m["days_to_close"] = (m.won_date - m.first_contact_date).dt.days
m["mm"] = m.first_contact_date.dt.to_period("M")

print("=== Q1. Win rate by MQL month (deals file only holds wins from Dec 2017) ===")
t = m.groupby("mm").agg(mql=("mql_id", "size"), won=("won", "sum"), win_rate=("won", "mean"), med_close=("days_to_close", "median"))
print(t.round(3).to_string())
early = m[m.first_contact_date < "2017-12-01"]
print(f"MQLs before Dec 2017: {len(early):,}, win rate {early.won.mean():.3f}; their wins all closed after {early.won_date.min()}")

coh = m[m.first_contact_date >= "2017-12-01"].copy()
print(f"\nclean MQL cohort (first contact Dec 2017 to May 2018): {len(coh):,}, win rate {coh.won.mean():.3f}")
print("days to close quantiles (cohort wins):", coh.days_to_close.quantile([.25, .5, .75, .9]).to_dict())

print("\n=== Q2. Lead decay ===")
for d in [0, 7, 14, 30, 60, 90]:
    alive = coh[~(coh.days_to_close < d)]
    print(f"not won by day {d:>2}: {len(alive):,} leads, eventually won {alive.won.mean():.3f}")

print("\n=== Q3. Lead scoring (train Dec 2017 to Mar 2018, test Apr to May 2018) ===")
coh["origin"] = coh.origin.fillna("unknown")
coh["dow"] = coh.first_contact_date.dt.dayofweek
tr, te = coh[coh.first_contact_date < "2018-04-01"].copy(), coh[coh.first_contact_date >= "2018-04-01"].copy()
prior = tr.won.mean()
stats = tr.groupby("landing_page_id").won.agg(["sum", "count"])
enc = (stats["sum"] + 20 * prior) / (stats["count"] + 20)
for d in (tr, te):
    d["page_rate"] = d.landing_page_id.map(enc).fillna(prior)
    d["page_n"] = d.landing_page_id.map(stats["count"]).fillna(0)
    d["origin_c"] = pd.Categorical(d.origin, categories=sorted(coh.origin.unique()))
print(f"train {len(tr):,} (win {tr.won.mean():.3f}) | test {len(te):,} (win {te.won.mean():.3f}); "
      f"test leads on pages unseen in train {(te.page_n == 0).mean():.3f}")
for name, F in [("origin only", ["origin_c"]), ("landing page only", ["page_rate", "page_n"]),
                ("origin + page + weekday", ["origin_c", "page_rate", "page_n", "dow"])]:
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                         random_state=42, categorical_features="from_dtype").fit(tr[F], tr.won)
    p = clf.predict_proba(te[F])[:, 1]
    top = p >= np.quantile(p, 0.8)
    print(f"{name:<26} AUC {roc_auc_score(te.won, p):.3f} | top 20% win rate {te.won.values[top].mean():.3f} vs {te.won.mean():.3f}")
print("win rate by origin (cohort):", coh.groupby("origin").won.agg(["mean", "size"]).sort_values("mean").round(3).to_dict("index"))

print("\n=== Q4. Sales rep effect on activation (deals won by 1 Jun 2018, first sale within 90 days) ===")
items = pd.read_csv(RAW / FILES["items"], usecols=["order_id", "seller_id"])
orders = pd.read_csv(RAW / FILES["orders"], usecols=["order_id", "order_purchase_timestamp"], parse_dates=["order_purchase_timestamp"])
first = items.merge(orders, on="order_id").groupby("seller_id").order_purchase_timestamp.min()
d = deals[deals.won_date <= "2018-06-01"].copy()
d["lag"] = (d.seller_id.map(first) - d.won_date).dt.days
d["act90"] = d.lag.between(0, 90).astype(int)
g = d.groupby("sr_id").act90.agg(["mean", "size"])
big = g[g["size"] >= 20]
print(f"deals {len(d)}, activation {d.act90.mean():.3f}; reps with 20+ deals: {len(big)}")
print(big.sort_values("mean").round(3).to_string())
sub = d[d.sr_id.isin(big.index)]
chi2, pv, dof, _ = chi2_contingency(pd.crosstab(sub.sr_id, sub.act90))
print(f"chi square across reps: chi2 {chi2:.1f}, dof {dof}, p {pv:.4f}")
for col in ["lead_type", "business_type", "business_segment"]:
    ct = pd.crosstab(sub.sr_id, sub[col].fillna("na"))
    chi2b, pvb, _, _ = chi2_contingency(ct)
    print(f"rep vs {col}: p {pvb:.4f} (reps get different lead mixes if small)")
# within lead type: does the rep spread survive?
sub = sub.assign(lt=sub.lead_type.fillna("na"))
exp = sub.groupby("lt").act90.transform("mean")
adj = sub.assign(exp=exp).groupby("sr_id").agg(actual=("act90", "mean"), expected=("exp", "mean"), n=("act90", "size"))
adj["lift"] = adj.actual - adj.expected
print("rep activation vs expected from lead type mix:\n" + adj.sort_values("lift").round(3).to_string())
