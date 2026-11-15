"""Walk forward training and out of sample scoring for the three window models.

Every prediction for month M comes from a model trained only on rows whose outcome was already known
before M started. That makes the whole history replayable: the app's as-of date shows exactly what the
system would have said on that day.

Labels follow the calendar date rule and count lost parcels and never dispatched orders. Training and evaluation use the
clean cohort. The dispatch window adds live seller signals, order windows get risk tiers by daily rank, and big sellers get
a recency score in the silence forecast."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from .config import (GBM_PARAMS, SILENCE_PARAMS, SCORE_FROM, SILENCE_SCORE_FROM, DATA_END, HIGH_SILENCE, TIER_HIGH,
                     TIER_ELEVATED, SURVEY_AFTER_PROMISE_D, BIG_SELLER_O90)
from .live import LIVE_SELLER

SELLER_HISTORY = ["life_orders", "tenure_d", "o_90", "late_dispatch_180", "late_dispatch_life_shrunk", "dispatch_n_180",
                  "late_delivery_180", "review_shrunk_180", "fail_rate_90"]
ORDER_NUMERIC = ["price", "freight", "n_items", "n_sellers", "weight_g", "photos", "desc_len", "dist_km", "same_state",
                 "promise_d", "limit_d", "dow", "hour", "moy", "installments", "new_seller"]
ORDER_CATEGORICAL = ["category", "customer_state", "seller_state", "pay_type"]
DISPATCH_FEATURES = ORDER_NUMERIC + ["s_" + c for c in SELLER_HISTORY] + ORDER_CATEGORICAL
CARRIER_FEATURES = DISPATCH_FEATURES + ["dispatch_delay_d"]
SILENCE_FEATURES = ["tenure_d", "recency_d", "life_orders", "o_0_30", "o_30_60", "o_60_90", "o_prev90", "g_90", "g_prev90",
                    "gmv_trend", "momentum", "active_m6", "peak_month_orders", "dispatch_n_180", "late_dispatch_180",
                    "overdue_now", "late_delivery_180", "reviews_n_180", "review_shrunk_180", "low_review_share_180",
                    "fail_rate_90", "avg_price_90", "freight_ratio_90", "n_categories_90", "dist_km_90", "moy"]


def _gbm(params):
    return HistGradientBoostingClassifier(**params, categorical_features="from_dtype")


def _auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(set(y)) > 1 else None


def _month_metrics(y, p, high_cut):
    y, p = np.asarray(y), np.asarray(p)
    top = p >= np.quantile(p, 0.9)
    high = p >= high_cut
    return dict(n=int(len(y)), base_rate=float(y.mean()), auc=_auc(y, p),
                top10_rate=float(y[top].mean()), high_n=int(high.sum()),
                high_rate=float(y[high].mean()) if high.any() else None)


def add_order_features(orders: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Attach seller history as of the previous month end (never the current month)."""
    o = orders.copy()
    o["prev_month"] = (o.ts_purchase.dt.to_period("M") - 1).astype(str)
    hist = panel[["seller_id", "month"] + SELLER_HISTORY].rename(columns={c: "s_" + c for c in SELLER_HISTORY})
    o = o.merge(hist.rename(columns={"month": "prev_month"}), on=["seller_id", "prev_month"], how="left")
    o["new_seller"] = o.s_life_orders.isna().astype(float)
    o["dow"] = o.ts_purchase.dt.dayofweek
    o["hour"] = o.ts_purchase.dt.hour
    o["moy"] = o.ts_purchase.dt.month
    for c in ORDER_CATEGORICAL:
        o[c] = o[c].fillna("unknown").astype("category")
    assert len(o) == len(orders), "feature join must not change the row count"
    return o


# Training rows must have started their window long enough before the scored month that almost all have resolved.
# Using "outcome already known" instead keeps fast orders and drops slow ones still in flight, biasing the model
# toward punctuality (measured: the carrier model then lost to a one line rule).
LAG_DAYS = {"late_dispatch": 30, "late_delivery": 60}
MIN_TRAIN = 3000   # months with less history than this are not scored


def score_order_window(o: pd.DataFrame, target: str, features: list, anchor: str, scored_rows, high_cut: float):
    """Reference only: one score per order at the start of its window (the console uses the daily models below)."""
    o = o.copy()
    o["p_" + target] = np.nan
    months = pd.period_range(SCORE_FROM, DATA_END.to_period("M"), freq="M")
    metrics = []
    for M in months:
        start = M.to_timestamp()
        cutoff = start - pd.Timedelta(days=LAG_DAYS[target])
        train = o[(o[anchor] < cutoff) & o[target].notna() & o.clean]
        if len(train) < MIN_TRAIN:
            continue
        test_idx = o.index[(o.ts_purchase.dt.to_period("M") == M) & scored_rows]
        if len(test_idx) == 0:
            continue
        model = _gbm(GBM_PARAMS).fit(train[features], train[target])
        o.loc[test_idx, "p_" + target] = model.predict_proba(o.loc[test_idx, features])[:, 1]
        test = o.loc[test_idx]
        test = test[test[target].notna() & test.clean]
        if len(test) > 200 and test[target].nunique() == 2:
            metrics.append(dict(month=str(M), train_n=int(len(train)), **_month_metrics(test[target], test["p_" + target], high_cut)))
    return o["p_" + target], pd.DataFrame(metrics)


def score_silence(panel: pd.DataFrame):
    """Silence forecast at horizons of 1, 2 and 3 months for sellers that sold in the last 90 days.

    Horizon k is trained on origins whose k month label window had closed (origin <= M - (k + 2)).
    Sellers with BIG_SELLER_O90+ orders in the last 90 days get a recency score instead of the model: the historical silence
    rate of big sellers with the same days since last order (one feature logistic fit on the same training rows). The model
    ranked big sellers worse than recency alone (audit S8 in analysis/iter2)."""
    P = panel[panel.o_90 > 0].copy()
    P["moy"] = pd.PeriodIndex(P.month, freq="M").month
    P["m"] = pd.PeriodIndex(P.month, freq="M")
    P["big_seller"] = P.o_90 >= BIG_SELLER_O90
    out = {k: pd.Series(np.nan, index=P.index) for k in (1, 2, 3)}
    raw = {k: pd.Series(np.nan, index=P.index) for k in (1, 2, 3)}
    metrics = []
    for M in pd.period_range(SILENCE_SCORE_FROM, DATA_END.to_period("M"), freq="M"):
        test_idx = P.index[P.m == M]
        for k in (1, 2, 3):
            y = f"silent_by_{k}"
            train = P[(P.m <= M - (k + 2)) & P[y].notna()]
            model = _gbm(SILENCE_PARAMS).fit(train[SILENCE_FEATURES], train[y])
            raw[k].loc[test_idx] = model.predict_proba(P.loc[test_idx, SILENCE_FEATURES])[:, 1]
            out[k].loc[test_idx] = raw[k].loc[test_idx]
            big_train = train[train.big_seller]
            big_test = test_idx[P.loc[test_idx, "big_seller"].values]
            if big_train[y].sum() >= 10 and len(big_test):
                # One feature logistic fit: strictly increasing in recency, so big sellers are ranked exactly by recency.
                lr = LogisticRegression().fit(np.log1p(big_train[["recency_d"]]), big_train[y])
                if lr.coef_[0, 0] > 0:
                    out[k].loc[big_test] = lr.predict_proba(np.log1p(P.loc[big_test, ["recency_d"]]))[:, 1]
            test = P.loc[test_idx]
            test = test[test[y].notna()]
            if len(test) > 100 and test[y].nunique() == 2:
                m = _month_metrics(test[y], out[k].loc[test.index], HIGH_SILENCE)
                m["auc_model_only"] = _auc(test[y], raw[k].loc[test.index])
                m["auc_recency_only"] = _auc(test[y], test.recency_d)
                metrics.append(dict(month=str(M), horizon=k, train_n=int(len(train)), **m))
    # A seller cannot be more likely silent by month 1 than by month 3: enforce nesting.
    p1 = out[1]
    p2 = np.maximum(out[2], p1)
    p3 = np.maximum(out[3], p2)
    scores = pd.DataFrame({"seller_id": P.seller_id, "month": P.month, "big_seller": P.big_seller,
                           "p_silent_1": p1, "p_silent_2": p2, "p_silent_3": p3})

    # Pooled accuracy by seller size (months from Sep 2017), so the big seller rule can be checked on enough sellers.
    by_size = []
    ev = P[P.m >= pd.Period("2017-09", "M")]
    for k in (1, 2, 3):
        y = f"silent_by_{k}"
        for size, g in [("10+ orders in 90 days", ev[ev.big_seller]), ("1 to 9 orders in 90 days", ev[~ev.big_seller])]:
            g = g[g[y].notna() & raw[k].loc[g.index].notna()]
            by_size.append(dict(horizon=k, size=size, n=int(len(g)), silent=int(g[y].sum()),
                                auc_model=_auc(g[y], raw[k].loc[g.index]), auc_recency=_auc(g[y], g.recency_d),
                                auc_shipped=_auc(g[y], out[k].loc[g.index])))
    return scores.dropna(subset=["p_silent_1"]), pd.DataFrame(metrics), pd.DataFrame(by_size)


# Daily snapshot models.
# An order sitting in a window is re-scored every day, because its risk changes as time passes: an order still not
# dispatched three days after purchase is riskier than the same order on day zero. Scoring only at the start of the
# window ranked live queues worse than "days already waiting" alone, so the console uses these snapshot scores.
WINDOWS = {
    # name: (label, anchor time, event that ends the window, deadline, base features)
    "dispatch": ("late_dispatch", "ts_purchase", "ts_carrier", "ship_limit", DISPATCH_FEATURES + LIVE_SELLER),
    "carrier": ("late_delivery", "ts_carrier", "ts_delivered", "ts_estimated",
                ["promise_d", "dispatch_delay_d", "customer_state", "seller_state", "dist_km", "moy", "freight", "weight_g"]),
}
# An order whose window ends without the event is known late once the deadline (plus the survey lag for parcels) passed.
GRACE_DAYS = {"dispatch": 0, "carrier": SURVEY_AFTER_PROMISE_D}
# For a parcel still in transit, fewer days left before the promise can only raise the risk. The constrained lean
# carrier model beat both the full feature set and the "days left" rule; live lane congestion signals did
# not add anything (feasibility test F2 in analysis/iter2), so the carrier features stay lean.
MONOTONE = {"carrier": {"days_left": -1}}
SNAPSHOT_EXTRA = ["days_waiting", "days_left"]


def build_snapshots(o: pd.DataFrame, window: str, live: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per order per day it is still inside the window at the end of that day."""
    label, anchor, event, deadline, feats = WINDOWS[window]
    order_feats = [f for f in feats if f not in LIVE_SELLER]
    base = o[o[anchor].notna() & o[deadline].notna() & o.seller_id.notna() & ~o.failed].copy()
    start = base[anchor].dt.normalize()
    span = ((base[deadline] - start).dt.days + 1).clip(lower=0, upper=60).astype(int)
    idx = np.repeat(base.index.values, span.values)
    k = np.concatenate([np.arange(n) for n in span.values])
    cols = list(dict.fromkeys(["order_id", "seller_id", "clean", anchor, event, deadline, label] + order_feats))
    snap = base.loc[idx, cols].reset_index(drop=True)
    snap["day"] = start.loc[idx].values + pd.to_timedelta(k + 1, unit="D") - pd.Timedelta(seconds=1)   # end of day
    still_open = snap[event].isna() | (snap[event] > snap.day)
    snap = snap[still_open & (snap[deadline] >= snap.day)].copy()
    snap["days_waiting"] = (snap.day - snap[anchor]).dt.total_seconds() / 86400
    snap["days_left"] = (snap[deadline] - snap.day).dt.total_seconds() / 86400
    snap["dday"] = snap.day.dt.normalize()
    if any(f in LIVE_SELLER for f in feats):
        assert live is not None, f"{window} needs the live seller signals"
        snap = snap.merge(live.rename(columns={"day": "dday"}), on=["seller_id", "dday"], how="left")
    return snap.reset_index(drop=True)


def score_snapshots(o: pd.DataFrame, window: str, train_fraction: float = 0.5, live: pd.DataFrame | None = None):
    """Monthly walk forward on daily snapshots. Returns daily scores with tiers, monthly metrics and order level metrics."""
    label, anchor, event, deadline, feats = WINDOWS[window]
    features = feats + SNAPSHOT_EXTRA
    snap = build_snapshots(o, window, live)
    snap["p"] = np.nan
    month = snap.day.dt.to_period("M")
    rng = np.random.default_rng(42)
    keep_for_training = rng.random(len(snap)) < train_fraction
    grace = pd.Timedelta(days=GRACE_DAYS[window])
    months = []
    for M in pd.period_range(SCORE_FROM, DATA_END.to_period("M"), freq="M"):
        test_idx = snap.index[month == M]
        if len(test_idx) == 0:
            continue
        s0 = M.to_timestamp()
        known = (snap[event] < s0) | (snap[event].isna() & (snap[deadline] + grace < s0))
        train = snap[keep_for_training & (snap[anchor] < s0 - pd.Timedelta(days=LAG_DAYS[label])) & known
                     & snap[label].notna() & snap.clean]
        if len(train) < MIN_TRAIN:
            continue
        model = HistGradientBoostingClassifier(**GBM_PARAMS, categorical_features="from_dtype",
                                               monotonic_cst=MONOTONE.get(window)).fit(train[features], train[label])
        snap.loc[test_idx, "p"] = model.predict_proba(snap.loc[test_idx, features])[:, 1]
        months.append((M, len(train)))

    # Tiers by rank within each day: High = riskiest 10% of the orders open that day, Elevated = the next 20%.
    scored = snap[snap.p.notna()].copy()
    rank = scored.groupby("dday").p.rank(pct=True, ascending=False, method="first")
    scored["tier"] = np.where(rank <= TIER_HIGH, 2, np.where(rank <= TIER_ELEVATED, 1, 0)).astype("int8")
    smonth = scored.day.dt.to_period("M")

    metrics, order_metrics = [], []
    for M, train_n in months:
        test = scored[(smonth == M) & scored[label].notna() & scored.clean]
        if len(test) <= 500 or test[label].nunique() < 2:
            continue
        y = test[label].values
        m = _month_metrics(y, test.p, np.inf)
        m.pop("high_n"), m.pop("high_rate")
        m["auc_days_waiting_only"] = _auc(y, test.days_waiting)
        m["auc_days_left_only"] = _auc(y, -test.days_left)
        for t, name in [(2, "high"), (1, "elevated"), (0, "normal")]:
            m[f"{name}_rate"] = float(y[test.tier.values == t].mean()) if (test.tier.values == t).any() else None
        metrics.append(dict(month=str(M), train_n=int(train_n), **m))
        # Order level: had the KAM worked the High tier every day of the month, which late orders would they have seen?
        g = test.assign(hi=test.tier.eq(2)).groupby("order_id").agg(y=(label, "first"), ever_high=("hi", "max"))
        first_high = test[test.tier == 2].groupby("order_id").days_left.max()
        caught = g[(g.y == 1) & g.ever_high]
        order_metrics.append(dict(month=str(M), orders=int(len(g)), late=int(g.y.sum()), flagged=int(g.ever_high.sum()),
                                  precision=float(g[g.ever_high].y.mean()) if g.ever_high.any() else None,
                                  recall=float(g[g.y == 1].ever_high.mean()) if (g.y == 1).any() else None,
                                  warning_days_median=float(first_high.reindex(caught.index).median()) if len(caught) else None))
    out = scored[["order_id", "day", "p", "tier"]].copy()
    out["day"] = out.day.dt.date
    out["p"] = out.p.astype("float32")
    return out, pd.DataFrame(metrics), pd.DataFrame(order_metrics)
