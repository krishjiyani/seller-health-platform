"""Feasibility F2 and F3: do corrected labels and live signals change the two order window models?

Variants per window (daily snapshot models, monthly walk forward, same parameters as iteration 1):
  V0 iteration 1 features, iteration 1 label (reference)
  V1 iteration 1 features, corrected label (late_or_lost / late_dispatch_full), clean cohort
  V2 V1 plus live signals (carrier: lane congestion; dispatch: seller backlog and recent lateness)
Reports AUC, calibration (mean predicted vs actual), Brier and top 10% hit rate by month, plus shock months.
Run: python -m analysis.iter2.feas_02_window_upgrades [dispatch|carrier]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from analysis.iter2.base import load, lane_day_stats, seller_day_stats
from pipeline.config import GBM_PARAMS
from pipeline.models import WINDOWS, LAG_DAYS
from pipeline.live import LIVE_SELLER

window = sys.argv[1] if len(sys.argv) > 1 else "carrier"
o, _ = load()
o = o[o.seller_id.notna()].copy()
days = pd.date_range("2017-01-01", "2018-09-15", freq="D")
_, anchor, event, deadline, feats = WINDOWS[window]
feats = [f for f in feats if f not in LIVE_SELLER]   # iteration 1 features; the pipeline now includes the live ones
old_label = "late_dispatch" if window == "dispatch" else "late_delivery"
new_label = "late_dispatch_full" if window == "dispatch" else "late_or_lost"
grace = 0 if window == "dispatch" else 2     # lost parcels are known late once promise + 2 has passed

if window == "carrier":
    lane, back = lane_day_stats(o, days)
    LIVE = ["lane_transit_med_30", "lane_late_share_30", "lane_n_30", "plat_transit_med_7", "dest_backlog"]
else:
    sd = seller_day_stats(o, days)
    LIVE = ["s_open_now", "s_late_14", "s_hand_n_14", "s_surge", "s_handover_med_30"]

base = o[o[anchor].notna() & o[deadline].notna() & ~o.failed].copy()
start = base[anchor].dt.normalize()
span = ((base[deadline] - start).dt.days + 1).clip(lower=0, upper=60).astype(int)
idx = np.repeat(base.index.values, span.values)
k = np.concatenate([np.arange(n) for n in span.values])
cols = ["order_id", "seller_id", "lane", "customer_state", anchor, event, deadline, old_label, new_label, "clean"] + feats
snap = base.loc[idx, list(dict.fromkeys(cols))].reset_index(drop=True)
snap["day"] = start.loc[idx].values + pd.to_timedelta(k + 1, unit="D") - pd.Timedelta(seconds=1)
snap = snap[(snap[event].isna() | (snap[event] > snap.day)) & (snap[deadline] >= snap.day)].reset_index(drop=True)
snap["days_waiting"] = (snap.day - snap[anchor]).dt.total_seconds() / 86400
snap["days_left"] = (snap[deadline] - snap.day).dt.total_seconds() / 86400
snap["dday"] = snap.day.dt.normalize()
if window == "carrier":
    snap = snap.merge(lane.rename(columns={"day": "dday"}), on=["lane", "dday"], how="left")
    snap = snap.merge(back.rename(columns={"day": "dday"}), on=["customer_state", "dday"], how="left")
else:
    snap = snap.merge(sd.rename(columns={"day": "dday"}), on=["seller_id", "dday"], how="left")
print(f"{window}: {len(snap):,} order days; live null share {snap[LIVE].isna().mean().round(3).to_dict()}")

X = feats + ["days_waiting", "days_left"]
VARIANTS = {"V0": (X, old_label, False), "V1": (X, new_label, True), "V2": (X + LIVE, new_label, True)}
mono = {"days_left": -1} if window == "carrier" else None
rng = np.random.default_rng(42)
keep = rng.random(len(snap)) < (0.5 if window == "dispatch" else 0.35)
month = snap.day.dt.to_period("M")
rows = []
for M in pd.period_range("2017-09", "2018-08", freq="M"):
    s0 = M.to_timestamp()
    test_all = snap[month == M]
    for v, (F, y, clean_only) in VARIANTS.items():
        known = (snap[event] < s0) | (snap[event].isna() & (snap[deadline] + pd.Timedelta(days=grace) < s0))
        tr = snap[keep & (snap[anchor] < s0 - pd.Timedelta(days=LAG_DAYS[old_label])) & known & snap[y].notna()]
        if clean_only:
            tr = tr[tr.clean]
        te = test_all[test_all[y].notna()]
        if clean_only:
            te = te[te.clean]
        m = HistGradientBoostingClassifier(**GBM_PARAMS, categorical_features="from_dtype",
                                           monotonic_cst=mono).fit(tr[F], tr[y])
        p = m.predict_proba(te[F])[:, 1]
        top = p >= np.quantile(p, 0.9)
        r = dict(month=str(M), variant=v, n=len(te), rate=te[y].mean(), mean_p=p.mean(), auc=roc_auc_score(te[y], p),
                 brier=brier_score_loss(te[y], p), top10=te[y].values[top].mean())
        if v == "V1":
            r["auc_rule"] = roc_auc_score(te[y], te.days_waiting if window == "dispatch" else -te.days_left)
        rows.append(r)
    print(M, "done", flush=True)

r = pd.DataFrame(rows)
print(f"\n=== {window.upper()}: per month ===")
print(r.pivot(index="month", columns="variant", values="auc").round(3).join(
      r[r.variant == "V1"].set_index("month")[["auc_rule"]].round(3)).to_string())
print("\ncalibration gap (mean predicted minus actual rate):")
print((r.assign(gap=r.mean_p - r.rate)).pivot(index="month", columns="variant", values="gap").round(3).to_string())
summary = r.groupby("variant")[["auc", "brier", "top10", "rate", "mean_p"]].mean()
summary["abs_calib_gap"] = r.assign(g=(r.mean_p - r.rate).abs()).groupby("variant").g.mean()
print("\nsummary (mean over months):\n" + summary.round(4).to_string())
print(f"rule AUC (V1 label): {r[r.variant == 'V1'].auc_rule.mean():.3f}")
r.to_csv(Path(__file__).parent / "out" / f"feas_02_{window}.csv", index=False)
