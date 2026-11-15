"""Feasibility F6: bad reviews that are not about lateness, and what customers complain about.

A. How are 1 to 2 star reviews split across delivery outcomes (on time, late, lost, never dispatched, cancelled)?
B. Order shape: do multi item and multi seller orders draw more bad reviews (the "only one item arrived" problem)?
C. Can a bad review on an order that will arrive ON TIME be predicted at purchase, with product and seller history?
D. Complaint taxonomy from review text (Portuguese keyword rules, no model): coverage and mix by outcome.
Run: python -m analysis.iter2.feas_06_reviews"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from analysis.iter2.base import load
from pipeline.build import RAW, FILES
from pipeline.models import DISPATCH_FEATURES

o, _ = load()
o = o[o.review_score.notna()].copy()
o["outcome"] = np.select([o.failed, o.never_dispatched, o.lost, o.late_date == 1, o.late_date == 0],
                         ["cancelled", "never dispatched", "lost in transit", "late", "on time"], "other")

print("=== A. Where the 1 to 2 star reviews sit ===")
a = o.groupby("outcome").agg(orders=("order_id", "size"), low_rate=("low_review", "mean"), low_n=("low_review", "sum"))
a["share_of_all_low"] = a.low_n / a.low_n.sum()
print(a.sort_values("low_n", ascending=False).round(3).to_string())

print("\n=== B. Order shape (on time deliveries only) ===")
ot = o[o.outcome == "on time"]
print(ot.groupby([ot.n_items.clip(upper=3), ot.n_sellers.clip(upper=2)]).low_review.agg(["mean", "size"]).round(3).to_string())

print("\n=== C. Predicting a bad review on an on time order, at purchase ===")
items = pd.read_csv(RAW / FILES["items"], usecols=["order_id", "product_id", "price"])
main = items.sort_values("price", ascending=False).drop_duplicates("order_id").set_index("order_id").product_id
o["product_id"] = o.order_id.map(main)
# product history: reviews of the same product written before this purchase
pr = o[["product_id", "ts_review", "low_review"]].dropna().sort_values("ts_review")
rv = pr.rename(columns={"ts_review": "t"})
rv["cum_low"] = rv.groupby("product_id").low_review.cumsum()
rv["cum_n"] = rv.groupby("product_id").cumcount() + 1
q = o[["order_id", "product_id", "ts_purchase"]].dropna().sort_values("ts_purchase")
m = pd.merge_asof(q, rv[["t", "product_id", "cum_low", "cum_n"]].sort_values("t"), left_on="ts_purchase", right_on="t",
                  by="product_id", allow_exact_matches=False)
prior = o.low_review.mean()
m["prod_low_shrunk"] = (m.cum_low.fillna(0) + 5 * prior) / (m.cum_n.fillna(0) + 5)
m["prod_n_reviews"] = m.cum_n.fillna(0)
o = o.merge(m[["order_id", "prod_low_shrunk", "prod_n_reviews"]], on="order_id", how="left")
o["freight_ratio"] = o.freight / o.price
o["prod_low_shrunk"] = o.prod_low_shrunk.fillna(prior)
F = DISPATCH_FEATURES + ["prod_low_shrunk", "prod_n_reviews", "freight_ratio"]
for c in ["category", "customer_state", "seller_state", "pay_type"]:
    o[c] = o[c].astype("category")
for scope, d in [("on time orders", o[o.outcome == "on time"]), ("all reviewed orders", o)]:
    d = d[d.clean]
    tr, te = d[(d.ts_purchase >= "2017-02-01") & (d.ts_purchase < "2018-01-01")], d[(d.ts_purchase >= "2018-01-01") & (d.ts_purchase < "2018-08-01")]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, random_state=42,
                                         categorical_features="from_dtype").fit(tr[F], tr.low_review)
    p = clf.predict_proba(te[F])[:, 1]
    top = p >= np.quantile(p, 0.9)
    print(f"{scope}: test rate {te.low_review.mean():.3f} | AUC {roc_auc_score(te.low_review, p):.3f} | "
          f"product history alone {roc_auc_score(te.low_review, te.prod_low_shrunk):.3f} | "
          f"seller history alone {roc_auc_score(te.low_review, -te.s_review_shrunk_180.fillna(te.s_review_shrunk_180.mean())):.3f} | "
          f"top 10% rate {te.low_review.values[top].mean():.3f}")

print("\n=== D. Complaint taxonomy (1 to 2 star reviews with text) ===")
TAGS = {
    "not received": r"n[aã]o (recebi|chegou|foi entregue|entreg)|ainda n[aã]o|nunca chegou|aguardando|n[aã]o veio",
    "late": r"atras|demor|prazo|lento",
    "partial or missing item": r"falt|apenas (um|uma|1)|s[oó] (um|uma|1|veio)|somente (um|uma|1)|incomplet|metade|outra parte|um dos|veio s[oó]",
    "wrong item": r"errad|diferente|outro produto|trocad|n[aã]o (era|é) o",
    "defect or quality": r"defeit|quebrad|danific|n[aã]o funciona|qualidade|estragad|rasgad|amassad|falsific|vazamento|riscad",
    "seller or service": r"atendimento|resposta|respond|contato|descaso|sac\b|reclama",
    "refund or cancel": r"devol|reembols|estorn|cancel|dinheiro de volta",
}
low = o[(o.low_review == 1) & o.review_text.notna()].copy()
t = low.review_text.str.lower()
for k, pat in TAGS.items():
    low[k] = t.str.contains(pat, regex=True)
low["any_tag"] = low[list(TAGS)].any(axis=1)
print(f"low reviews {int(o.low_review.sum()):,}; with text {len(low):,} ({len(low) / o.low_review.sum():.1%}); tagged by at least one rule {low.any_tag.mean():.1%}")
print(low.groupby("outcome")[list(TAGS) + ["any_tag"]].mean().round(3).T.to_string())
print("orders per outcome (low with text):", low.outcome.value_counts().to_dict())
ot_low = low[low.outcome == "on time"]
print(f"on time and low: 'not received' {ot_low['not received'].mean():.1%}; of those, multi item {ot_low[ot_low['not received']].n_items.gt(1).mean():.1%}, "
      f"multi seller {ot_low[ot_low['not received']].n_sellers.gt(1).mean():.1%}")
seller_fault = ot_low[["partial or missing item", "wrong item", "defect or quality"]].any(axis=1).mean()
print(f"on time and low: tagged as product or fulfilment problem (partial, wrong, defect) {seller_fault:.1%}")
