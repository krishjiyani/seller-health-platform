"""Builds every file the app reads. Usage: python -m pipeline.run"""
import json
import time
import pandas as pd
from . import build, models
from .config import APP, HIGH_DISPATCH, HIGH_CARRIER


def main():
    t0 = time.time()
    APP.mkdir(parents=True, exist_ok=True)
    con = build.load_raw()
    print(f"raw tables loaded and checked ({time.time() - t0:.0f}s)")
    orders = build.build_orders(con)
    panel = build.build_seller_months(orders)
    mql, deals = build.build_onboarding(con, orders)
    print(f"orders {len(orders):,} | seller months {len(panel):,} | deals {len(deals)} ({time.time() - t0:.0f}s)")

    o = models.add_order_features(orders, panel)
    has_items = o.seller_id.notna() & o.ship_limit.notna()
    o["p_late_dispatch"], m_dispatch = models.score_order_window(
        o, "late_dispatch", models.DISPATCH_FEATURES, anchor="ts_purchase", scored_rows=has_items, high_cut=HIGH_DISPATCH)
    o["p_late_delivery"], m_carrier = models.score_order_window(
        o, "late_delivery", models.CARRIER_FEATURES, anchor="ts_carrier", scored_rows=has_items & o.ts_carrier.notna(), high_cut=HIGH_CARRIER)
    print(f"order windows scored at window start ({time.time() - t0:.0f}s)")
    snapshot_metrics = {}
    for window in models.WINDOWS:
        daily, snapshot_metrics[window] = models.score_snapshots(o, window)
        daily.to_parquet(APP / f"daily_{window}.parquet", index=False)
        print(f"{window} daily snapshots: {len(daily):,} rows ({time.time() - t0:.0f}s)")
    silence, m_silence = models.score_silence(panel)
    print(f"silence forecast scored ({time.time() - t0:.0f}s)")

    order_cols = ["order_id", "seller_id", "status", "ts_purchase", "ts_approved", "ts_carrier", "ts_delivered", "ts_estimated",
                  "ship_limit", "price", "freight", "category", "customer_state", "seller_state", "pay_type", "review_score",
                  "review_text", "ts_review", "late_dispatch", "late_delivery", "low_review", "failed", "dispatch_delay_d",
                  "unjust_review", "st_approved", "st_dispatched_on_time", "st_delivered_on_time", "st_good_review",
                  "p_late_dispatch", "p_late_delivery"]
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
    metrics = {"dispatch": m_dispatch.to_dict("records"), "carrier": m_carrier.to_dict("records"),
               "silence": m_silence.to_dict("records"),
               "dispatch_daily": snapshot_metrics["dispatch"].to_dict("records"),
               "carrier_daily": snapshot_metrics["carrier"].to_dict("records"), "built_seconds": round(time.time() - t0)}
    (APP / "metrics.json").write_text(json.dumps(metrics, indent=1))

    for name, m in [("dispatch", m_dispatch), ("carrier", m_carrier)]:
        print(f"{name}: mean AUC {m.auc.mean():.3f} over {len(m)} months | base {m.base_rate.mean():.3f} | high flag hit rate {m.high_rate.mean():.3f}")
    for name, m in snapshot_metrics.items():
        print(f"{name} daily: mean AUC {m.auc.mean():.3f} | days waiting only {m.auc_days_waiting_only.mean():.3f} | "
              f"days left only {m.auc_days_left_only.mean():.3f} | high flag hit rate {m.high_rate.mean():.3f}")
    for k, g in m_silence.groupby("horizon"):
        print(f"silence +{k}m: mean AUC {g.auc.mean():.3f} (recency only {g.auc_recency_only.mean():.3f}) over {len(g)} months")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
