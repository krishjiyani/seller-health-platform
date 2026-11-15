"""Builds every file the app reads. Usage: python -m pipeline.run"""
import json
import time
import pandas as pd
from . import build, models
from .config import APP, HIGH_DISPATCH, HIGH_CARRIER, DATA_START, DATA_END
from .live import seller_day_stats
from .taxonomy import TAG_COLS


def main():
    t0 = time.time()
    APP.mkdir(parents=True, exist_ok=True)
    con = build.load_raw()
    print(f"raw tables loaded and checked ({time.time() - t0:.0f}s)")
    orders = build.build_orders(con)
    panel = build.build_seller_months(orders)
    mql, deals = build.build_onboarding(con, orders)
    first_sale, ramp_rule = build.build_new_sellers(con, orders)
    print(f"orders {len(orders):,} | seller months {len(panel):,} | deals {len(deals)} ({time.time() - t0:.0f}s)")

    o = models.add_order_features(orders, panel)
    has_items = o.seller_id.notna() & o.ship_limit.notna()
    o["p_late_dispatch"], m_dispatch = models.score_order_window(
        o, "late_dispatch", models.DISPATCH_FEATURES, anchor="ts_purchase", scored_rows=has_items, high_cut=HIGH_DISPATCH)
    o["p_late_delivery"], m_carrier = models.score_order_window(
        o, "late_delivery", models.CARRIER_FEATURES, anchor="ts_carrier", scored_rows=has_items & o.ts_carrier.notna(), high_cut=HIGH_CARRIER)
    print(f"order windows scored at window start (reference) ({time.time() - t0:.0f}s)")
    live = seller_day_stats(o, pd.date_range(DATA_START, DATA_END.normalize() + pd.Timedelta(days=1), freq="D"))
    print(f"live seller signals: {len(live):,} seller days ({time.time() - t0:.0f}s)")
    snapshot_metrics, order_metrics = {}, {}
    for window in models.WINDOWS:
        daily, snapshot_metrics[window], order_metrics[window] = models.score_snapshots(o, window, live=live)
        daily.to_parquet(APP / f"daily_{window}.parquet", index=False)
        print(f"{window} daily snapshots: {len(daily):,} rows ({time.time() - t0:.0f}s)")
    silence, m_silence, silence_by_size = models.score_silence(panel)
    print(f"silence forecast scored ({time.time() - t0:.0f}s)")

    order_cols = ["order_id", "seller_id", "status", "ts_purchase", "ts_approved", "ts_carrier", "ts_delivered", "ts_estimated",
                  "ship_limit", "survey_due", "price", "freight", "n_items", "n_sellers", "category", "customer_state", "seller_state",
                  "pay_type", "review_score", "review_text", "ts_review", "late_dispatch", "late_delivery", "low_review", "failed",
                  "lost", "never_dispatched", "clean", "dispatch_delay_d", "unjust_review", "bad_review_cause",
                  "st_approved", "st_dispatched_on_time", "st_delivered_on_time", "st_good_review",
                  "p_late_dispatch", "p_late_delivery"] + list(TAG_COLS.values())
    out = o[order_cols].copy()
    for c in ["category", "customer_state", "seller_state", "pay_type"]:
        out[c] = out[c].astype(str)
    out.to_parquet(APP / "orders.parquet", index=False)
    panel_cols = ["seller_id", "month", "life_orders", "tenure_d", "recency_d", "o_0_30", "o_30_60", "o_60_90", "o_90", "g_90",
                  "o_prev90", "g_prev90", "gmv_trend", "momentum", "late_dispatch_180", "overdue_now", "late_delivery_180",
                  "review_shrunk_180", "low_review_share_180", "fail_rate_90", "silent_by_1", "silent_by_2", "silent_by_3", "gmv_next_90"]
    panel[panel_cols].merge(silence, on=["seller_id", "month"], how="left").to_parquet(APP / "seller_months.parquet", index=False)
    mql.to_parquet(APP / "mql.parquet", index=False)
    deals.to_parquet(APP / "deals.parquet", index=False)
    first_sale.to_parquet(APP / "sellers.parquet", index=False)
    rec = lambda df: df.to_dict("records")
    metrics = {"dispatch": rec(m_dispatch), "carrier": rec(m_carrier), "silence": rec(m_silence),
               "silence_by_size": rec(silence_by_size),
               "dispatch_daily": rec(snapshot_metrics["dispatch"]), "carrier_daily": rec(snapshot_metrics["carrier"]),
               "dispatch_orders": rec(order_metrics["dispatch"]), "carrier_orders": rec(order_metrics["carrier"]),
               "ramp_rule": rec(ramp_rule), "built_seconds": round(time.time() - t0)}
    (APP / "metrics.json").write_text(json.dumps(metrics, indent=1))

    for name, m in [("dispatch", m_dispatch), ("carrier", m_carrier)]:
        print(f"{name} (window start): mean AUC {m.auc.mean():.3f} over {len(m)} months | base {m.base_rate.mean():.3f}")
    for name, m in snapshot_metrics.items():
        e = m[m.month >= "2017-09"]
        beat = (e.auc > (e.auc_days_waiting_only if name == "dispatch" else e.auc_days_left_only)).sum()
        print(f"{name} daily (Sep 2017 on): mean AUC {e.auc.mean():.3f} | days waiting only {e.auc_days_waiting_only.mean():.3f} | "
              f"days left only {e.auc_days_left_only.mean():.3f} | beats the rule in {beat} of {len(e)} months | "
              f"High tier late {e.high_rate.mean():.3f} vs Normal {e.normal_rate.mean():.3f}")
        om = order_metrics[name]
        om = om[om.month >= "2017-09"]
        print(f"   orders: High tier caught {om.recall.mean():.3f} of late orders, precision {om.precision.mean():.3f}, "
              f"median warning {om.warning_days_median.mean():.1f} days")
    for k, g in m_silence[m_silence.month >= "2017-09"].groupby("horizon"):
        print(f"silence +{k}m: mean AUC {g.auc.mean():.3f} (model only {g.auc_model_only.mean():.3f}, recency only "
              f"{g.auc_recency_only.mean():.3f}) over {len(g)} months")
    print(silence_by_size.round(3).to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
