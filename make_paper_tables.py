"""Build paper tables from data/app/metrics.json, the pipeline's own output.

Run from the repo root after `python -m pipeline.run`:
    python make_paper_tables.py
Writes LaTeX tables to paper_tables/ and prints a text summary. Needs only pandas and numpy.
Handles the older metrics.json (4 window keys + silence) and the newer one (adds flag precision/recall,
silence_by_size, tier rates, ramp_rule); sections missing from the file are skipped.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
FROM = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else "2017-09"   # same burn-in as run.py headline numbers
SRC, OUT = Path("data/app/metrics.json"), Path("paper_tables")
OUT.mkdir(exist_ok=True)
m = json.load(open(SRC))


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def tex(header, rows, cols):
    body = [" & ".join(r) + r" \\" for r in rows]
    return "\n".join([r"\begin{tabular}{" + cols + "}", r"\hline", " & ".join(header) + r" \\", r"\hline", *body,
                      r"\hline", r"\end{tabular}"])


def windows():
    for k in ("dispatch", "carrier", "dispatch_daily", "carrier_daily"):
        if k in m:
            d = pd.DataFrame(m[k])
            yield k, d[d.month >= FROM].reset_index(drop=True)
    if "silence" in m:
        s = pd.DataFrame(m["silence"])
        for h, g in s[s.month >= FROM].groupby("horizon"):
            yield f"silence_{h}m", g.reset_index(drop=True)


def rules(d):   # simple single-input rules only; auc_model_only is the model itself, not a rule
    return {c[4:].replace("_only", "").replace("_", " "): d[c] for c in d.columns
            if c.startswith("auc_") and "model" not in c}


summary = []
for name, d in windows():
    r = rules(d)
    rn, rs = max(r.items(), key=lambda kv: kv[1].mean()) if r else ("n/a", None)   # strongest rule by mean AUC
    summary.append(dict(window=name, folds=len(d), first=d.month.iloc[0], last=d.month.iloc[-1], base=d.base_rate.mean(),
                        auc=d.auc.mean(), sd=d.auc.std(), rule=rn, rule_auc=rs.mean() if rs is not None else np.nan,
                        wins=f"{int((d.auc > rs).sum())}/{len(d)}" if rs is not None else "n/a",
                        model_only=d.auc_model_only.mean() if "auc_model_only" in d else np.nan))
    cols = [c for c in d.columns if c in ("month", "n", "base_rate", "auc", "top10_rate", "high_rate", "elevated_rate",
                                           "normal_rate") or c.startswith("auc_")]
    (OUT / f"fold_{name}.tex").write_text(tex([esc(c) for c in cols],
                                              [[str(v) for v in x] for x in d[cols].round(3).itertuples(index=False)], "l" * len(cols)))
S = pd.DataFrame(summary)
print(S.round(3).to_string(index=False))
f3 = lambda v: "n/a" if pd.isna(v) else f"{v:.3f}"
(OUT / "summary.tex").write_text(tex(
    ["Window", "Folds", "Test months", "Base rate", "AUC", "Strongest rule", "Rule AUC", "Months AUC > rule"],
    [[esc(r.window), str(r.folds), f"{r['first']} to {r['last']}", f3(r.base), f"{r.auc:.3f} $\\pm$ {r.sd:.3f}", esc(r.rule),
      f3(r.rule_auc), r.wins] for _, r in S.iterrows()], "lllllllc"))

for k in ("dispatch_orders", "carrier_orders"):   # operating point: what a KAM would actually see
    if k in m:
        d = pd.DataFrame(m[k])
        d = d[d.month >= FROM]
        print(f"\n{k}: mean precision {d.precision.mean():.3f}, recall {d.recall.mean():.3f}, "
              f"median warning {d.warning_days_median.median():.2f} days, flagged share {(d.flagged.sum() / d.orders.sum()):.3f}")
        (OUT / f"flags_{k}.tex").write_text(tex(
            ["Month", "Orders", "Late", "Flagged", "Precision", "Recall", "Median warning (d)"],
            [[str(r.month), str(int(r.orders)), str(int(r.late)), str(int(r.flagged)), f"{r.precision:.3f}", f"{r.recall:.3f}",
              f"{r.warning_days_median:.1f}"] for r in d.itertuples()], "lrrrrrr"))
if "silence_by_size" in m:
    d = pd.DataFrame(m["silence_by_size"]).round(3)
    print("\n", d.to_string(index=False))
    (OUT / "silence_by_size.tex").write_text(tex([esc(c) for c in d.columns],
                                                 [[str(v) for v in r] for r in d.itertuples(index=False)], "l" * len(d.columns)))
if "ramp_rule" in m:
    d = pd.DataFrame(m["ramp_rule"]).round(3)
    (OUT / "ramp_rule.tex").write_text(tex([esc(c) for c in d.columns],
                                           [[str(v) for v in r] for r in d.itertuples(index=False)], "lrr"))
print(f"\nTables written to {OUT}/")
