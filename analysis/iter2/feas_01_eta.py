"""Feasibility F1: delivery time (ETA) prediction and a promise policy test.

Q1. At purchase, how well can actual purchase to delivery days be predicted (P50), versus a lane median and versus Olist's promise?
Q2. If the promise were set from a predicted quantile, what late rate and promise length would follow, versus Olist's?
Q3. At handover, how well can the remaining days to delivery be predicted (for proactive customer messages in the carrier window)?
Walk forward monthly, Sep 2017 to Aug 2018, clean cohort, training rows anchored at least 60 days before the test month.
Run: python -m analysis.iter2.feas_01_eta"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from analysis.iter2.base import load, lane_day_stats, seller_day_stats
from pipeline.models import DISPATCH_FEATURES

o, _ = load()
o = o[o.seller_id.notna() & o.clean].copy()
days = pd.date_range("2017-01-01", "2018-09-15", freq="D")
lane, back = lane_day_stats(o, days)
sd = seller_day_stats(o, days)
o["pday"] = o.ts_purchase.dt.normalize()
o["cday"] = o.ts_carrier.dt.normalize()
o = o.merge(lane.rename(columns={"day": "pday"}), on=["lane", "pday"], how="left")
o = o.merge(back.rename(columns={"day": "pday"}), on=["customer_state", "pday"], how="left")
o = o.merge(sd[["seller_id", "day", "s_handover_med_30", "s_open_now", "s_late_14"]].rename(columns={"day": "pday"}), on=["seller_id", "pday"], how="left")
LIVE = ["lane_transit_med_30", "lane_late_share_30", "lane_n_30", "plat_transit_med_7", "dest_backlog", "s_handover_med_30", "s_open_now", "s_late_14"]
FEAT = DISPATCH_FEATURES + LIVE
for c in ["category", "customer_state", "seller_state", "pay_type"]:
    o[c] = o[c].astype("category")
print(f"orders {len(o):,}; live features null share:", o[LIVE].isna().mean().round(3).to_dict())

QS = [0.5, 0.8, 0.9, 0.95]
res, pol = [], []
o["pred50"] = np.nan
for M in pd.period_range("2017-09", "2018-08", freq="M"):
    start = M.to_timestamp()
    train = o[(o.ts_purchase < start - pd.Timedelta(days=60)) & o.actual_d.notna()]
    test = o[(o.ts_purchase.dt.to_period("M") == M) & o.actual_d.notna()].copy()
    if len(test) < 200:
        continue
    for q in QS:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=40, random_state=42, categorical_features="from_dtype")
        test[f"q{int(q * 100)}"] = m.fit(train[FEAT], train.actual_d).predict(test[FEAT])
    # baseline: lane median of purchase to delivery days over the 60 days before the train cutoff window ends (known at test time)
    hist = o[(o.ts_delivered < start) & (o.ts_delivered >= start - pd.Timedelta(days=60))]
    lm = hist.groupby("lane", observed=True).actual_d.median()
    test["lane_med"] = test.lane.map(lm).astype(float).fillna(hist.actual_d.median())
    o.loc[test.index, "pred50"] = test.q50
    res.append(dict(month=str(M), n=len(test), mae_model=(test.q50 - test.actual_d).abs().mean(),
                    mae_lane_median=(test.lane_med - test.actual_d).abs().mean(),
                    mae_olist_promise=(test.promise_d - test.actual_d).abs().mean(),
                    cover_q90=(test.actual_d <= test.q90).mean(), cover_q95=(test.actual_d <= test.q95).mean()))
    deliv_day = test.ts_delivered.dt.normalize()
    row = dict(month=str(M), olist_late=(deliv_day > test.ts_estimated).mean(), olist_promise_d=test.promise_d.mean())
    for q in [80, 90, 95]:
        promised = test.pday + pd.to_timedelta(np.ceil(test[f"q{q}"]), unit="D")
        row[f"q{q}_late"] = (deliv_day > promised).mean()
        row[f"q{q}_promise_d"] = np.ceil(test[f"q{q}"]).mean()
    pol.append(row)

res, pol = pd.DataFrame(res), pd.DataFrame(pol)
print("\n=== Q1. ETA at purchase (days, P50 model) ===")
print(res.round(3).to_string(index=False))
print("mean:", res.drop(columns=["month"]).mean().round(3).to_dict())
print("\n=== Q2. Promise policy: late rate and mean promised days ===")
print(pol.round(3).to_string(index=False))
print("mean:", pol.drop(columns=["month"]).mean().round(3).to_dict())

# Q3. Remaining days at handover (carrier window ETA).
print("\n=== Q3. Remaining days to delivery at handover ===")
h = o[o.ts_carrier.notna() & o.ts_delivered.notna() & (o.transit_d >= 0)].copy()
h = h.drop(columns=LIVE[:5]).merge(lane.rename(columns={"day": "cday"}), on=["lane", "cday"], how="left")
h = h.merge(back.rename(columns={"day": "cday"}), on=["customer_state", "cday"], how="left")
h["days_left_at_handover"] = (h.ts_estimated - h.ts_carrier).dt.total_seconds() / 86400
CF = ["promise_d", "dispatch_delay_d", "days_left_at_handover", "customer_state", "seller_state", "dist_km", "moy", "freight",
      "weight_g", "lane_transit_med_30", "lane_late_share_30", "lane_n_30", "plat_transit_med_7", "dest_backlog"]
CF_BASE = CF[:9]
r3 = []
for M in pd.period_range("2017-09", "2018-08", freq="M"):
    start = M.to_timestamp()
    train = h[h.ts_carrier < start - pd.Timedelta(days=60)]
    test = h[h.ts_carrier.dt.to_period("M") == M].copy()
    if len(test) < 200:
        continue
    out = dict(month=str(M), n=len(test))
    for name, feats in [("base", CF_BASE), ("live", CF)]:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=40, random_state=42, categorical_features="from_dtype")
        p = m.fit(train[feats], train.transit_d).predict(test[feats])
        out[f"mae_{name}"] = np.abs(p - test.transit_d).mean()
        if name == "live":
            pdate = (test.ts_carrier + pd.to_timedelta(p, unit="D")).dt.normalize()
            pred_late = pdate > test.ts_estimated
            out["late_flag_precision"] = test.late_date[pred_late].mean() if pred_late.any() else np.nan
            out["late_flag_recall"] = pred_late[test.late_date == 1].mean()
    lm = h[(h.ts_delivered < start) & (h.ts_delivered >= start - pd.Timedelta(days=60))].groupby("lane", observed=True).transit_d.median()
    out["mae_lane_median"] = np.abs(test.lane.map(lm).astype(float).fillna(test.transit_d.median()) - test.transit_d).mean()
    r3.append(out)
r3 = pd.DataFrame(r3)
print(r3.round(3).to_string(index=False))
print("mean:", r3.drop(columns=["month"]).mean().round(3).to_dict())
