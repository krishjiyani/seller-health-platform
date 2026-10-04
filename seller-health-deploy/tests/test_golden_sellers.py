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


def main():
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
