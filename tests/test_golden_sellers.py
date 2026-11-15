"""Golden seller check: recompute a few sellers' panel values straight from the raw CSVs with plain pandas
(an independent code path from the DuckDB pipeline) and require an exact match.
Run after the pipeline: python -m tests.test_golden_sellers"""
import pandas as pd
from pipeline.config import RAW, APP

AS_OF = "2018-04"
T = pd.Period(AS_OF, "M").to_timestamp(how="end")


def recompute(seller_id: str) -> dict:
    items = pd.read_csv(RAW / "olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
    orders = pd.read_csv(RAW / "olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp", "order_delivered_carrier_date"])
    # Attribution rule: highest item value seller per order, ties to lowest seller_id.
    v = items.groupby(["order_id", "seller_id"]).price.sum().reset_index().sort_values(["order_id", "price", "seller_id"], ascending=[True, False, True])
    owner = v.drop_duplicates("order_id").set_index("order_id").seller_id
    lim = items.groupby("order_id").shipping_limit_date.max()
    price = items.groupby("order_id").price.sum()
    mine = orders[orders.order_id.map(owner) == seller_id].copy()
    mine = mine[(mine.order_purchase_timestamp >= "2017-01-01") & (mine.order_purchase_timestamp <= T)]
    w90 = mine[mine.order_purchase_timestamp > T - pd.Timedelta(days=90)]
    ho = mine[(mine.order_delivered_carrier_date <= T) & (mine.order_purchase_timestamp > T - pd.Timedelta(days=180))]
    late = (ho.order_delivered_carrier_date > ho.order_id.map(lim))
    return dict(life_orders=len(mine), o_90=len(w90), g_90=round(w90.order_id.map(price).sum(), 2),
                late_dispatch_180=round(late.mean(), 6) if len(ho) else None)


def check_label_definitions():
    """Label definition checks on the app tables, recomputed from the raw orders file."""
    raw = pd.read_csv(RAW / "olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp", "order_delivered_customer_date",
                                                                      "order_estimated_delivery_date"])
    raw = raw[(raw.order_purchase_timestamp >= "2017-01-01") & (raw.order_purchase_timestamp <= "2018-08-31 23:59:59")]
    d = raw[raw.order_delivered_customer_date.notna()]
    late = (d.order_delivered_customer_date.dt.normalize() > d.order_estimated_delivery_date).mean()
    assert 0.065 < late < 0.070, f"late share of delivered orders by calendar date: {late:.4f}"
    o = pd.read_parquet(APP / "orders.parquet")
    app_late = o[o.ts_delivered.notna()].late_delivery.mean()
    assert abs(app_late - late) < 1e-9, f"app late share {app_late:.4f} differs from raw {late:.4f}"
    dv = o[o.ts_delivered.notna() & o.ts_review.notna()]
    survey = (dv.ts_review.dt.normalize() == dv.survey_due).mean()
    assert survey > 0.90, f"review survey rule matches {survey:.3f}"
    assert (o.late_delivery[o.lost] == 1).all() and (o.late_dispatch[o.never_dispatched] == 1).all()
    print(f"label definitions: late share {late:.2%} (raw and app agree), survey rule matches {survey:.1%}, "
          f"{int(o.lost.sum()):,} lost and {int(o.never_dispatched.sum()):,} never dispatched counted as late")


def main():
    check_label_definitions()
    panel = pd.read_parquet(APP / "seller_months.parquet")
    p = panel[panel.month == AS_OF].set_index("seller_id")
    golden = list(p.sort_values("o_90", ascending=False).index[[0, 25, 200]])   # a large, a mid and a small seller
    for sid in golden:
        expected = recompute(sid)
        got = dict(life_orders=int(p.loc[sid, "life_orders"]), o_90=int(p.loc[sid, "o_90"]), g_90=round(float(p.loc[sid, "g_90"]), 2),
                   late_dispatch_180=round(float(p.loc[sid, "late_dispatch_180"]), 6) if pd.notna(p.loc[sid, "late_dispatch_180"]) else None)
        assert got == expected, f"{sid[:8]}: pipeline {got} vs independent {expected}"
        print(f"{sid[:8]} matches: {got}")
    print("golden sellers: all match")


if __name__ == "__main__":
    main()
