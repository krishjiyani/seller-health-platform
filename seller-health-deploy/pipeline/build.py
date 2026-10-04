"""Raw CSVs to clean, app ready tables. Every step ends with assertions.

Rules documented here and in the README:
* Order attribution: an order belongs to the seller with the highest item value in it (ties: lowest seller_id). 1.3% of orders have several sellers.
* Review dedupe: one review per order, the one with the latest answer timestamp.
* Dispatch deadline: the latest shipping_limit_date across the order's items.
"""
import duckdb
import numpy as np
import pandas as pd
from .config import RAW, DATA_START, DATA_END, SHRINK_C

FILES = {
    "orders": "olist_orders_dataset.csv", "items": "olist_order_items_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv", "payments": "olist_order_payments_dataset.csv",
    "customers": "olist_customers_dataset.csv", "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv", "geo": "olist_geolocation_dataset.csv",
    "cat_tr": "product_category_name_translation.csv",
    "mql": "olist_marketing_qualified_leads_dataset.csv", "deals": "olist_closed_deals_dataset.csv",
}
EXPECTED_ROWS = dict(orders=99441, items=112650, reviews=99224, payments=103886, customers=99441, products=32951,
                     sellers=3095, geo=1000163, cat_tr=71, mql=8000, deals=842)


def load_raw() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for name, f in FILES.items():
        con.execute(f"create table {name} as select * from read_csv_auto('{RAW / f}', header=true)")
        n = con.execute(f"select count(*) from {name}").fetchone()[0]
        assert n == EXPECTED_ROWS[name], f"{name}: {n} rows, expected {EXPECTED_ROWS[name]}"
    for child, parent, key in [("items", "orders", "order_id"), ("items", "sellers", "seller_id"),
                               ("items", "products", "product_id"), ("orders", "customers", "customer_id"),
                               ("deals", "mql", "mql_id")]:
        orphans = con.execute(f"select count(*) from {child} anti join {parent} using({key})").fetchone()[0]
        assert orphans == 0, f"{child}->{parent}: {orphans} orphans"
    return con


def build_orders(con) -> pd.DataFrame:
    """One row per order with timestamps, outcome flags, attributed seller and context."""
    df = con.execute("""
    with seller_val as (
        select order_id, seller_id, sum(price) v from items group by 1, 2),
    main_seller as (
        select order_id, arg_min(seller_id, (-v, seller_id)) seller_id from seller_val group by 1),
    item_agg as (
        select i.order_id, sum(price) price, sum(freight_value) freight, count(*) n_items,
               count(distinct i.seller_id) n_sellers, max(shipping_limit_date) ship_limit,
               sum(coalesce(p.product_weight_g, 0)) weight_g, avg(p.product_photos_qty) photos,
               avg(p.product_description_lenght) desc_len,
               arg_max(coalesce(t.product_category_name_english, p.product_category_name, 'unknown'), price) category
        from items i left join products p using(product_id) left join cat_tr t using(product_category_name)
        group by 1),
    rev as (
        select order_id, arg_max(review_score, (review_answer_timestamp, review_id)) review_score,
               arg_max(review_comment_message, (review_answer_timestamp, review_id)) review_text,
               arg_max(review_creation_date, (review_answer_timestamp, review_id)) ts_review
        from reviews group by 1),
    pay as (
        select order_id, arg_max(payment_type, payment_value) pay_type, max(payment_installments) installments
        from payments group by 1),
    geo_c as (
        select geolocation_zip_code_prefix zip, avg(geolocation_lat) lat, avg(geolocation_lng) lng from geo group by 1)
    select o.order_id, o.order_status status,
           o.order_purchase_timestamp ts_purchase, o.order_approved_at ts_approved,
           o.order_delivered_carrier_date ts_carrier, o.order_delivered_customer_date ts_delivered,
           o.order_estimated_delivery_date ts_estimated, a.ship_limit,
           m.seller_id, a.price, a.freight, a.n_items, a.n_sellers, a.weight_g, a.photos, a.desc_len, a.category,
           c.customer_state, s.seller_state, r.review_score, r.review_text, r.ts_review, p.pay_type, p.installments,
           2 * 6371 * asin(sqrt(pow(sin(radians(gc.lat - gs.lat) / 2), 2)
               + cos(radians(gs.lat)) * cos(radians(gc.lat)) * pow(sin(radians(gc.lng - gs.lng) / 2), 2))) dist_km
    from orders o
    left join main_seller m using(order_id)
    left join item_agg a using(order_id)
    left join customers c using(customer_id)
    left join sellers s on s.seller_id = m.seller_id
    left join rev r using(order_id)
    left join pay p using(order_id)
    left join geo_c gc on gc.zip = c.customer_zip_code_prefix
    left join geo_c gs on gs.zip = s.seller_zip_code_prefix
    """).df()
    assert len(df) == 99441 and df.order_id.is_unique
    df = df[(df.ts_purchase >= DATA_START) & (df.ts_purchase <= DATA_END)].copy()

    # Outcome flags. NaN means "not observable" (e.g. never handed over), never "on time".
    df["late_dispatch"] = (df.ts_carrier > df.ship_limit).astype(float).where(df.ts_carrier.notna() & df.ship_limit.notna())
    df["late_delivery"] = (df.ts_delivered > df.ts_estimated).astype(float).where(df.ts_delivered.notna())
    df["low_review"] = (df.review_score <= 2).astype(float).where(df.review_score.notna())
    df["failed"] = df.status.isin(["canceled", "unavailable"])
    df["dispatch_delay_d"] = (df.ts_carrier - df.ship_limit).dt.total_seconds() / 86400
    df["promise_d"] = (df.ts_estimated - df.ts_purchase).dt.total_seconds() / 86400
    df["limit_d"] = (df.ship_limit - df.ts_purchase).dt.total_seconds() / 86400
    df["same_state"] = (df.customer_state == df.seller_state).astype(float)

    # Strictly conditional order journey: each stage requires the previous one.
    df["st_approved"] = df.ts_approved.notna()
    df["st_dispatched_on_time"] = df.st_approved & (df.late_dispatch == 0)
    df["st_delivered_on_time"] = df.st_dispatched_on_time & (df.late_delivery == 0)
    df["st_good_review"] = df.st_delivered_on_time & (df.review_score >= 4)

    # "Unjust" bad review: 1 or 2 stars on an order the seller dispatched on time but the carrier delivered late.
    df["unjust_review"] = (df.low_review == 1) & (df.late_dispatch == 0) & (df.late_delivery == 1)

    assert df.order_id.is_unique
    assert df.seller_id.isna().sum() <= 800, "orders without items should be about 775"
    assert 0.08 < df.late_dispatch.mean() < 0.10, df.late_dispatch.mean()
    assert 0.07 < df.late_delivery.mean() < 0.09, df.late_delivery.mean()
    assert (df.st_good_review <= df.st_delivered_on_time).all() and (df.st_delivered_on_time <= df.st_dispatched_on_time).all()
    return df


def build_seller_months(orders: pd.DataFrame) -> pd.DataFrame:
    """Seller history at every month end T, using only information timestamped on or before T.

    Also attaches forward labels (from months after T) used to train and validate the silence forecast."""
    o = orders[orders.seller_id.notna()].copy()
    o["m"] = o.ts_purchase.dt.to_period("M")
    months = pd.period_range(DATA_START, DATA_END, freq="M")
    counts = o.groupby(["seller_id", "m"]).size().unstack(fill_value=0).reindex(columns=months, fill_value=0)
    gmv = o.groupby(["seller_id", "m"]).price.sum().unstack(fill_value=0).reindex(columns=months, fill_value=0)
    A = counts.values
    prior_late = o.late_dispatch.mean()
    prior_review = o.review_score.mean()
    days = lambda n: pd.Timedelta(days=n)

    frames = []
    for i, M in enumerate(months):
        T = M.to_timestamp(how="end")
        h = o[o.ts_purchase <= T]
        g = h.groupby("seller_id")
        f = pd.DataFrame({"first_order": g.ts_purchase.min(), "last_order": g.ts_purchase.max(), "life_orders": g.size()})
        f["tenure_d"] = (T - f.first_order).dt.days
        f["recency_d"] = (T - f.last_order).dt.days
        for lo, hi, name in [(0, 30, "0_30"), (30, 60, "30_60"), (60, 90, "60_90"), (90, 180, "prev90")]:
            w = h[(h.ts_purchase > T - days(hi)) & (h.ts_purchase <= T - days(lo))].groupby("seller_id")
            f["o_" + name] = w.size()
            f["g_" + name] = w.price.sum()
        f = f.fillna({c: 0 for c in f.columns if c[:2] in ("o_", "g_")})
        f["o_90"] = f.o_0_30 + f.o_30_60 + f.o_60_90
        f["g_90"] = f.g_0_30 + f.g_30_60 + f.g_60_90
        f["gmv_trend"] = (f.g_90 - f.g_prev90) / (f.g_90 + f.g_prev90 + 1)
        f["momentum"] = (f.o_0_30 - f.o_60_90) / (f.o_90 + 1)
        f["active_m6"] = pd.Series((A[:, max(0, i - 5):i + 1] > 0).sum(1), index=counts.index)
        f["peak_month_orders"] = pd.Series(A[:, :i + 1].max(1), index=counts.index)

        # Dispatch reliability: only handovers that had happened by T.
        ho = h[h.ts_carrier <= T]
        recent = ho[ho.ts_purchase > T - days(180)].groupby("seller_id").late_dispatch
        f["dispatch_n_180"] = recent.count()
        f["late_dispatch_180"] = recent.mean()
        life = ho.groupby("seller_id").late_dispatch
        f["late_dispatch_life_shrunk"] = (SHRINK_C * prior_late + life.sum()) / (SHRINK_C + life.count())
        overdue = h[(h.ship_limit < T) & (h.ts_carrier.isna() | (h.ts_carrier > T)) & ~h.failed & (h.ts_purchase > T - days(90))]
        f["overdue_now"] = overdue.groupby("seller_id").size()
        dl = h[(h.ts_delivered <= T) & (h.ts_purchase > T - days(180))]
        f["late_delivery_180"] = dl.groupby("seller_id").late_delivery.mean()
        rv = h[(h.ts_review <= T) & (h.ts_purchase > T - days(180))].groupby("seller_id").review_score
        f["reviews_n_180"] = rv.count()
        f["review_shrunk_180"] = (SHRINK_C * prior_review + rv.sum()) / (SHRINK_C + rv.count())
        lowr = h[(h.ts_review <= T) & (h.ts_purchase > T - days(180))].groupby("seller_id").low_review
        f["low_review_share_180"] = lowr.mean()
        cx = h[(h.ts_purchase > T - days(90)) & (h.ts_purchase <= T - days(7))].groupby("seller_id").failed
        f["fail_rate_90"] = cx.mean()
        w90 = h[h.ts_purchase > T - days(90)]
        f["avg_price_90"] = w90.groupby("seller_id").price.mean()
        f["freight_ratio_90"] = (w90.freight / w90.price).groupby(w90.seller_id).mean()
        f["n_categories_90"] = w90.groupby("seller_id").category.nunique()
        f["dist_km_90"] = w90.groupby("seller_id").dist_km.mean()

        fill0 = ["dispatch_n_180", "overdue_now", "reviews_n_180"]
        f[fill0] = f[fill0].fillna(0)
        f["review_shrunk_180"] = f.review_shrunk_180.fillna(prior_review)
        f["late_dispatch_life_shrunk"] = f.late_dispatch_life_shrunk.fillna(prior_late)

        # Forward labels: "silent from month s" means months s, s+1, s+2 have zero orders (needs s+2 observed).
        rows = counts.index.get_indexer(f.index)
        def silent_from(s):
            if s + 2 >= len(months):
                return np.full(len(rows), np.nan)
            return (A[rows, s:s + 3].sum(1) == 0).astype(float)
        cum = None
        for k in (1, 2, 3):
            s = silent_from(i + k)
            cum = s if cum is None else np.where(np.isnan(s) | np.isnan(cum), np.nan, np.maximum(cum, s))
            f[f"silent_by_{k}"] = cum
        f["gmv_next_90"] = gmv.values[rows, i + 1:i + 4].sum(1) if i + 3 < len(months) else np.nan
        f["month"] = M
        frames.append(f.reset_index(names="seller_id"))

    panel = pd.concat(frames, ignore_index=True)
    panel["month"] = panel.month.astype(str)
    assert not panel.duplicated(["seller_id", "month"]).any()
    assert (panel.recency_d >= 0).all() and panel.late_dispatch_180.dropna().between(0, 1).all()
    both = panel[["silent_by_1", "silent_by_3"]].dropna()
    assert (both.silent_by_1 <= both.silent_by_3).all(), "silence horizons must be nested"
    return panel


def build_onboarding(con, orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lead funnel (MQL) and won deals with first sale and month one survival."""
    mql = con.execute("select * from mql").df()
    deals = con.execute("select d.*, m.origin from deals d join mql m using(mql_id)").df()
    mql["won"] = mql.mql_id.isin(deals.mql_id)
    o = orders[orders.seller_id.notna()]
    # First sale counts any order containing the seller's item, including orders attributed to a co-seller.
    first = con.execute(f"""
        select i.seller_id, min(o.order_purchase_timestamp) first_sale from items i join orders o using(order_id)
        where o.order_purchase_timestamp between '{DATA_START}' and '{DATA_END}' group by 1""").df().set_index("seller_id").first_sale
    deals = deals.merge(first, left_on="seller_id", right_index=True, how="left")
    deals["days_to_first_sale"] = (deals.first_sale - deals.won_date).dt.days
    om = o.assign(m=o.ts_purchase.dt.to_period("M")).groupby("seller_id").m.apply(set)
    def m1(row):
        if pd.isna(row.first_sale):
            return np.nan
        nxt = row.first_sale.to_period("M") + 1
        if nxt > DATA_END.to_period("M"):
            return np.nan
        return float(nxt in om.get(row.seller_id, set()))
    deals["active_month_after_first"] = deals.apply(m1, axis=1)
    keep = ["mql_id", "seller_id", "won_date", "business_segment", "lead_type", "business_type", "origin",
            "sr_id", "first_sale", "days_to_first_sale", "active_month_after_first"]
    deals = deals[keep]
    assert len(deals) == 842 and deals.seller_id.is_unique
    assert deals.first_sale.notna().sum() == 380, deals.first_sale.notna().sum()
    assert (deals.days_to_first_sale.dropna() >= 0).all()
    return mql[["mql_id", "first_contact_date", "origin", "won"]], deals
