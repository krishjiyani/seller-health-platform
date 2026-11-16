"""Extra analyses for the paper. Put in analysis/ and run from the repo root:

    python analysis/paper_extras.py            # Part A and Part B
    python analysis/paper_extras.py --skip-b --skip-c   # Part A only (needs just data/app/*.parquet)
    add  --from 2017-04  to include the burn-in months (default: 2017-09, as in the pipeline headline numbers)

Part A (reads the pipeline's saved out-of-sample scores, no retraining):
  1. ROC-AUC, PR-AUC, Brier score and Brier skill per window, next to simple-rule baselines
  2. Risk-tier validity: observed event rate per tier (daily windows)
  3. Silence queue: risk-only vs risk x revenue vs revenue-only ranking
Part B (needs data/raw/ and the pipeline code): grouped permutation importance of the silence model by health
dimension. This measures how much AUC is lost when a group of features is shuffled together on held-out months.
It is NOT an ablation (the model is not retrained) and it is not causal.

Part C (needs data/raw/ and the pipeline code): an untuned Logistic Regression baseline for the silence windows, next to
the gradient-boosted model and the recency rule, under the same walk-forward training rule as the pipeline.

Nothing here is tuned; every number comes from scores the pipeline already produced out of sample.
Writes LaTeX tables and CSVs to paper_tables/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
APP, OUT = ROOT / "data" / "app", ROOT / "paper_tables"
OUT.mkdir(exist_ok=True)
# Same burn-in as the pipeline's headline numbers (run.py reports Sep 2017 onward). Change with: --from 2017-04
FROM = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else "2017-09"

# Health-dimension grouping of the silence features. This grouping is defined here for analysis,
# it is not something the pipeline itself uses. Part B asserts that it covers SILENCE_FEATURES exactly.
GROUPS = {
    "Recency (days since last order)": ["recency_d"],
    "Commercial momentum (other activity)": ["o_0_30", "o_30_60", "o_60_90", "o_prev90", "g_90", "g_prev90", "gmv_trend",
                                             "momentum", "active_m6", "peak_month_orders"],
    "Dispatch and delivery": ["dispatch_n_180", "late_dispatch_180", "overdue_now", "late_delivery_180"],
    "Customer experience": ["reviews_n_180", "review_shrunk_180", "low_review_share_180"],
    "Reliability (cancellations)": ["fail_rate_90"],
    "Seller profile and mix": ["tenure_d", "life_orders", "avg_price_90", "freight_ratio_90", "n_categories_90", "dist_km_90", "moy"],
}


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def write_tex(name, header, rows, cols):
    body = [" & ".join(r) + r" \\" for r in rows]
    txt = "\n".join([r"\begin{tabular}{" + cols + "}", r"\hline", " & ".join(header) + r" \\", r"\hline", *body,
                     r"\hline", r"\end{tabular}"])
    (OUT / name).write_text(txt)


def month_metrics(df, y, model, rules, month_col, min_n=100):
    """Per-month ROC-AUC, PR-AUC and Brier for the model, and ROC-AUC/PR-AUC for each rule score (higher = riskier)."""
    rows = []
    for M, g in df.groupby(month_col):
        yy = g[y].to_numpy()
        if str(M) < FROM or len(g) < min_n or len(set(yy)) < 2:
            continue
        base = yy.mean()
        r = dict(month=str(M), n=len(g), base=base, roc=roc_auc_score(yy, g[model]),
                 pr=average_precision_score(yy, g[model]), brier=brier_score_loss(yy, g[model]))
        r["brier_skill"] = 1 - r["brier"] / (base * (1 - base))   # versus always predicting that month's base rate
        for name, col in rules.items():
            r["roc_" + name] = roc_auc_score(yy, g[col])
            r["pr_" + name] = average_precision_score(yy, g[col])
        rows.append(r)
    return pd.DataFrame(rows)


def daily_frame(daily, o, label, anchor, deadline):
    j = daily.merge(o[["order_id", label, anchor, deadline]], on="order_id", how="left")
    j = j[j[label].notna()].copy()
    end = pd.to_datetime(j["day"]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    j["days_waiting"] = (end - j[anchor]).dt.total_seconds() / 86400
    j["neg_days_left"] = -(j[deadline] - end).dt.total_seconds() / 86400
    j["month"] = pd.to_datetime(j["day"]).dt.to_period("M")
    return j


def part_a():
    o = pd.read_parquet(APP / "orders.parquet")
    sm = pd.read_parquet(APP / "seller_months.parquet")
    results = {}
    for k in (1, 2, 3):
        d = sm[sm[f"p_silent_{k}"].notna() & sm[f"silent_by_{k}"].notna()]
        results[f"silence +{k}m"] = month_metrics(d, f"silent_by_{k}", f"p_silent_{k}", {"recency": "recency_d"}, "month")
    tiers = {}
    for name, label, anchor, deadline in [("dispatch", "late_dispatch", "ts_purchase", "ship_limit"),
                                          ("carrier", "late_delivery", "ts_carrier", "ts_estimated")]:
        j = daily_frame(pd.read_parquet(APP / f"daily_{name}.parquet"), o, label, anchor, deadline)
        results[f"{name} daily"] = month_metrics(j, label, "p", {"days_waiting": "days_waiting", "days_left": "neg_days_left"},
                                                 "month", min_n=500)
        if "tier" in j:
            j = j[j.month.astype(str) >= FROM]
            t = j.groupby("tier")[label].agg(rows="size", rate="mean")
            t["share"] = t.rows / t.rows.sum()
            by_month = j.pivot_table(index="month", columns="tier", values=label, aggfunc="mean")
            ordered = (by_month.diff(axis=1).iloc[:, 1:] > 0).all(axis=1)   # each higher tier has a higher rate
            tiers[name] = (t, int(ordered.sum()), len(by_month))

    rows, tex_rows = [], []
    for w, d in results.items():
        rule_cols = [c[3:] for c in d.columns if c.startswith("pr_")]
        best = max(rule_cols, key=lambda r: d["pr_" + r].mean())
        row = dict(window=w, folds=len(d), base=d.base.mean(), roc=d.roc.mean(), pr=d.pr.mean(), best_rule=best,
                   roc_rule=d["roc_" + best].mean(), pr_rule=d["pr_" + best].mean(), brier=d.brier.mean(),
                   brier_skill=d.brier_skill.mean())
        rows.append(row)
        tex_rows.append([esc(w), str(row["folds"]), f"{row['base']:.3f}", f"{row['roc']:.3f}", f"{row['pr']:.3f}",
                         f"{row['pr_rule']:.3f} ({esc(best)})", f"{row['brier']:.3f}", f"{row['brier_skill']:.3f}"])
    S = pd.DataFrame(rows)
    print("\nA1. Mean over test months\n", S.round(3).to_string(index=False))
    S.to_csv(OUT / "extras_metrics.csv", index=False)
    write_tex("extras_metrics.tex", ["Window", "Folds", "Base rate", "ROC-AUC", "PR-AUC", "PR-AUC best rule", "Brier", "Brier skill"],
              tex_rows, "lccccccc")

    if tiers:
        tex_rows = []
        for w, (t, ok, n) in tiers.items():
            print(f"\nA2. {w} daily tiers (months with rate strictly increasing across tiers: {ok}/{n})\n", t.round(3).to_string())
            for tier, r in t.iterrows():
                tex_rows.append([w, str(tier), f"{int(r.rows):,}", f"{r.share:.3f}", f"{r.rate:.3f}"])
        write_tex("extras_tiers.tex", ["Window", "Tier (0 low to 2 high)", "Order-days", "Share", "Observed late rate"], tex_rows, "llrrr")

    for k in (3,):
        d = sm[sm[f"p_silent_{k}"].notna() & sm[f"silent_by_{k}"].notna()].copy()
        d["y"], d["risk"] = d[f"silent_by_{k}"], d[f"p_silent_{k}"]
        d["lost"], d["prio"] = d.g_90 * d.y, d.risk * d.g_90
        q_rows = []
        for M, g in d.groupby("month"):
            if str(M) < FROM or g.y.sum() == 0 or len(g) < 100:
                continue
            for q in (0.05, 0.10, 0.20):
                n = int(np.ceil(q * len(g)))
                for name, col in [("risk only", "risk"), ("risk x revenue", "prio"), ("revenue only", "g_90")]:
                    top = g.nlargest(n, col)
                    q_rows.append(dict(month=M, queue=q, ranking=name, silent_caught=top.y.sum() / g.y.sum(),
                                       revenue_caught=top.lost.sum() / g.lost.sum()))
        Q = pd.DataFrame(q_rows).groupby(["queue", "ranking"])[["silent_caught", "revenue_caught"]].mean().reset_index()
        print(f"\nA3. Silence within {k} months: share of silent sellers / of their 90-day GMV caught by the top-q queue "
              f"(random = q; mean over {len(set(r['month'] for r in q_rows))} months)\n", Q.round(3).to_string(index=False))
        Q.to_csv(OUT / "extras_queue.csv", index=False)
        write_tex("extras_queue.tex", ["Queue size", "Ranking", "Silent sellers caught", "Lost GMV caught"],
                  [[f"top {r.queue:.0%}", r.ranking, f"{r.silent_caught:.3f}", f"{r.revenue_caught:.3f}"] for r in Q.itertuples()], "llcc")


def group_permutation_importance(model, X, y, groups, n_repeats=10, seed=42):
    """AUC lost when all columns of a group are shuffled together (same row permutation for every column in the group)."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        rows.append(dict(group=name, auc_drop=float(np.mean(drops)), sd=float(np.std(drops))))
    return base, pd.DataFrame(rows).sort_values("auc_drop", ascending=False)


_CACHE = {}


def silence_panel():
    """Seller-month panel for the silence window, built once from data/raw with the pipeline's own code."""
    if "P" not in _CACHE:
        from pipeline import build, models
        feats = models.SILENCE_FEATURES
        flat = [c for cols in GROUPS.values() for c in cols]
        assert sorted(flat) == sorted(feats), f"Update GROUPS. Missing: {set(feats) - set(flat)}; extra: {set(flat) - set(feats)}"
        panel = build.build_seller_months(build.build_orders(build.load_raw()))
        P = panel[panel.o_90 > 0].copy()
        P["m"] = pd.PeriodIndex(P.month, freq="M")
        P["moy"] = P.m.dt.month
        _CACHE["P"], _CACHE["feats"] = P, feats
    return _CACHE["P"], _CACHE["feats"]


def silence_baselines(P, feats, make_gbm, from_month, horizons=(1, 2, 3), min_train=500, min_test=100):
    """Per test month: ROC-AUC and PR-AUC of an untuned Logistic Regression, the gradient-boosted model and the recency rule.

    Training rule as in the pipeline: horizon k trains on origins up to M-(k+2), so every label window has closed.
    The gradient-boosted scores here are the raw model scores (before the pipeline's nesting rule)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows = []
    for k in horizons:
        y = f"silent_by_{k}"
        for M in sorted(P.m.unique()):
            if str(M) < from_month:
                continue
            test = P[(P.m == M) & P[y].notna()]
            train = P[(P.m <= M - (k + 2)) & P[y].notna()]
            if len(test) < min_test or test[y].nunique() < 2 or len(train) < min_train or train[y].nunique() < 2:
                continue
            lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
            p_lr = lr.fit(train[feats], train[y]).predict_proba(test[feats])[:, 1]
            p_gb = make_gbm().fit(train[feats], train[y]).predict_proba(test[feats])[:, 1]
            yy = test[y].to_numpy()
            rows.append(dict(horizon=k, month=str(M), n=len(test), base=yy.mean(),
                             roc_lr=roc_auc_score(yy, p_lr), roc_gbm=roc_auc_score(yy, p_gb), roc_recency=roc_auc_score(yy, test.recency_d),
                             pr_lr=average_precision_score(yy, p_lr), pr_gbm=average_precision_score(yy, p_gb),
                             pr_recency=average_precision_score(yy, test.recency_d)))
    return pd.DataFrame(rows)


def part_c():
    if not (ROOT / "data" / "raw").exists():
        print("\nPart C skipped: data/raw/ not found.")
        return
    from pipeline import config, models
    P, feats = silence_panel()
    R = silence_baselines(P, feats, lambda: models._gbm(config.SILENCE_PARAMS), FROM)
    S = R.groupby("horizon")[["roc_lr", "roc_gbm", "roc_recency", "pr_lr", "pr_gbm", "pr_recency"]].mean()
    S.insert(0, "folds", R.groupby("horizon").size())
    print(f"\nC. Silence: untuned Logistic Regression vs gradient boosting vs recency (mean over test months from {FROM})\n", S.round(3).to_string())
    R.to_csv(OUT / "extras_logistic_by_month.csv", index=False)
    S.round(3).to_csv(OUT / "extras_logistic.csv")
    write_tex("extras_logistic.tex", ["Horizon", "Folds", "AUC LR", "AUC GBM", "AUC recency", "PR-AUC LR", "PR-AUC GBM", "PR-AUC recency"],
              [[f"{int(h)} mo.", str(int(r.folds)), f"{r.roc_lr:.3f}", f"{r.roc_gbm:.3f}", f"{r.roc_recency:.3f}", f"{r.pr_lr:.3f}",
                f"{r.pr_gbm:.3f}", f"{r.pr_recency:.3f}"] for h, r in S.iterrows()], "lccccccc")


def part_b():
    if not (ROOT / "data" / "raw").exists():
        print("\nPart B skipped: data/raw/ not found.")
        return
    from pipeline import config, models
    P, feats = silence_panel()
    tex_rows = []
    for k in (1, 3):
        y = f"silent_by_{k}"
        lab = P[P[y].notna()]
        test_months = [lab.m.max() - 2, lab.m.max() - 1, lab.m.max()]
        test = lab[lab.m.isin(test_months)]
        train = lab[lab.m <= min(test_months) - (k + 2)]   # same label-maturity rule as the pipeline
        model = models._gbm(config.SILENCE_PARAMS).fit(train[feats], train[y])
        base, imp = group_permutation_importance(model, test[feats], test[y], GROUPS)
        print(f"\nB. Silence +{k}m, test origins {test_months[0]} to {test_months[-1]} (n={len(test)}), "
              f"train n={len(train)}, AUC {base:.3f}\n", imp.round(4).to_string(index=False))
        imp.to_csv(OUT / f"extras_importance_{k}m.csv", index=False)
        for r in imp.itertuples():
            tex_rows.append([f"+{k}m", esc(r.group), f"{r.auc_drop:.3f} $\\pm$ {r.sd:.3f}"])
    write_tex("extras_importance.tex", ["Horizon", "Feature group shuffled", "AUC lost"], tex_rows, "llc")


if __name__ == "__main__":
    part_a()
    if "--skip-b" not in sys.argv:
        part_b()
    if "--skip-c" not in sys.argv:
        part_c()
    print(f"\nTest months from {FROM}. Tables written to {OUT}/")
