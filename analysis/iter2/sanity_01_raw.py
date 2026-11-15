"""Data sanity v2, part 1: raw table checks (nulls, ranges, timestamp order, cross table consistency).

Run: python -m analysis.iter2.sanity_01_raw  (from the repo root)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.common import connect

con = connect()
q = lambda s: con.execute(s).df()
one = lambda s: con.execute(s).fetchone()


def h(t):
    print(f"\n=== {t} ===")


h("ORDERS: status x missing timestamps")
print(q("""select order_status, count(*) n,
  sum((order_approved_at is null)::int) no_approved, sum((order_delivered_carrier_date is null)::int) no_carrier,
  sum((order_delivered_customer_date is null)::int) no_delivered
  from orders group by 1 order by 2 desc""").to_string())

h("ORDERS: purchase month counts at the edges")
print(q("""select strftime(order_purchase_timestamp, '%Y-%m') m, count(*) n from orders group by 1 order by 1""").to_string())

h("ORDERS: timestamp ordering violations")
print(q("""select
  sum((order_approved_at < order_purchase_timestamp)::int) approved_before_purchase,
  sum((order_delivered_carrier_date < order_purchase_timestamp)::int) carrier_before_purchase,
  sum((order_delivered_carrier_date < order_approved_at)::int) carrier_before_approved,
  sum((order_delivered_customer_date < order_delivered_carrier_date)::int) delivered_before_carrier,
  sum((order_delivered_customer_date < order_purchase_timestamp)::int) delivered_before_purchase,
  sum((order_estimated_delivery_date < order_purchase_timestamp)::int) estimate_before_purchase,
  sum((order_status = 'delivered' and order_delivered_customer_date is null)::int) delivered_status_no_date,
  sum((order_status in ('canceled','unavailable') and order_delivered_customer_date is not null)::int) cancelled_but_delivered,
  sum((order_status not in ('delivered','canceled') and order_delivered_customer_date is not null)::int) other_status_with_delivery
  from orders""").T.to_string())

h("ORDERS: carrier before approval, size of gap")
print(q("""select count(*) n, median(epoch(order_approved_at - order_delivered_carrier_date)/3600) median_gap_h,
  quantile_cont(epoch(order_approved_at - order_delivered_carrier_date)/3600, 0.9) p90_gap_h
  from orders where order_delivered_carrier_date < order_approved_at""").to_string())

h("ORDERS: time of day of estimated date (is it a date, not a timestamp?)")
print(q("""select (strftime(order_estimated_delivery_date, '%H:%M:%S') = '00:00:00') midnight, count(*) from orders group by 1""").to_string())

h("ORDERS: late delivery definition, timestamp vs calendar date")
print(q("""select count(*) delivered,
  avg((order_delivered_customer_date > order_estimated_delivery_date)::int) late_ts_rule,
  avg((order_delivered_customer_date::date > order_estimated_delivery_date::date)::int) late_date_rule,
  sum((order_delivered_customer_date::date = order_estimated_delivery_date::date)::int) delivered_on_promise_day
  from orders where order_delivered_customer_date is not null
  and order_purchase_timestamp between '2017-01-01' and '2018-08-31 23:59:59'""").T.to_string())

h("ORDERS: delivery and promise durations (days)")
print(q("""select
  quantile_cont(epoch(order_delivered_customer_date - order_purchase_timestamp)/86400, [0.1,0.5,0.9,0.99]) actual_purchase_to_delivery,
  quantile_cont(epoch(order_estimated_delivery_date - order_purchase_timestamp)/86400, [0.1,0.5,0.9,0.99]) promised_purchase_to_estimate,
  quantile_cont(epoch(order_delivered_customer_date - order_delivered_carrier_date)/86400, [0.1,0.5,0.9,0.99]) transit,
  quantile_cont(epoch(order_estimated_delivery_date - order_delivered_customer_date)/86400, [0.1,0.5,0.9]) slack_days_early
  from orders where order_delivered_customer_date is not null""").T.to_string())
print(q("""select count(*) extreme_transit_over_60d from orders
  where epoch(order_delivered_customer_date - order_delivered_carrier_date)/86400 > 60""").to_string())

h("ITEMS: shipping limit checks")
print(q("""with o as (select order_id, order_purchase_timestamp p, order_delivered_carrier_date c from orders)
  select count(*) items,
  sum((shipping_limit_date < p)::int) limit_before_purchase,
  sum((epoch(shipping_limit_date - p)/86400 > 30)::int) limit_over_30d_after_purchase,
  quantile_cont(epoch(shipping_limit_date - p)/86400, [0.1,0.5,0.9,0.99]) limit_days_q
  from items join o using(order_id)""").T.to_string())
print(q("""select count(*) orders_with_several_limits from (select order_id from items group by 1 having count(distinct shipping_limit_date) > 1)""").to_string())
print(q("""select count(*) orders_with_several_sellers from (select order_id from items group by 1 having count(distinct seller_id) > 1)""").to_string())

h("ITEMS: price and freight")
print(q("""select min(price) min_price, sum((price <= 0)::int) nonpositive_price, sum((freight_value = 0)::int) zero_freight,
  sum((freight_value > price)::int) freight_above_price, quantile_cont(price, [0.5, 0.99]) price_q, max(price) max_price from items""").T.to_string())

h("PAYMENTS: totals vs item totals")
print(q("""with p as (select order_id, sum(payment_value) pay, max(payment_installments) inst,
            bool_or(payment_type = 'voucher') has_voucher, count(*) n_rows from payments group by 1),
  i as (select order_id, sum(price + freight_value) due from items group by 1)
  select count(*) orders_both, sum((abs(pay - due) > 1)::int) mismatch_over_1brl,
  sum((pay - due > 1)::int) paid_more, sum((due - pay > 1)::int) paid_less,
  sum((abs(pay - due) > 1 and has_voucher)::int) mismatch_with_voucher
  from p join i using(order_id)""").T.to_string())
print(q("""select payment_type, count(*) n, sum((payment_value = 0)::int) zero_value, sum((payment_installments = 0)::int) zero_inst from payments group by 1""").to_string())
print(q("""select (select count(*) from orders anti join payments using(order_id)) orders_without_payment,
  (select count(*) from orders anti join items using(order_id)) orders_without_items""").to_string())

h("REVIEWS: duplicates and timing")
print(q("""select count(*) n_rows, count(distinct review_id) uniq_review_id, count(distinct order_id) uniq_orders,
  (select count(*) from (select order_id from reviews group by 1 having count(*) > 1)) orders_multi_review,
  (select count(*) from (select review_id from reviews group by 1 having count(distinct order_id) > 1)) review_id_on_many_orders
  from reviews""").T.to_string())
print(q("""select (strftime(review_creation_date, '%H:%M:%S') = '00:00:00') midnight, count(*) from reviews group by 1""").to_string())
print(q("""select o.order_status, count(*) n, avg(review_score) avg_score,
  avg((review_score <= 2)::int) low_share,
  sum((r.review_creation_date < o.order_purchase_timestamp::date)::int) review_before_purchase,
  sum((r.review_answer_timestamp < r.review_creation_date)::int) answer_before_creation
  from reviews r join orders o using(order_id) group by 1 order by 2 desc""").to_string())
print(q("""select (select count(*) from orders anti join reviews using(order_id)) orders_without_review,
  avg((review_comment_message is not null)::int) share_with_text, avg((review_comment_title is not null)::int) share_with_title from reviews""").to_string())

h("REVIEWS: creation date relative to delivery and promise (delivered orders)")
print(q("""select
  avg((review_creation_date < order_delivered_customer_date::date)::int) before_delivery_day,
  avg((review_creation_date = order_delivered_customer_date::date)::int) on_delivery_day,
  avg((review_creation_date < order_estimated_delivery_date::date and order_delivered_customer_date > order_estimated_delivery_date)::int) late_and_before_promise,
  quantile_cont(epoch(review_creation_date - order_delivered_customer_date::date)/86400, [0.1,0.5,0.9]) days_after_delivery_q
  from reviews r join orders o using(order_id) where order_delivered_customer_date is not null""").T.to_string())

h("CUSTOMERS")
print(q("""select count(*) customer_ids, count(distinct customer_unique_id) people,
  (select count(*) from (select customer_unique_id from customers group by 1 having count(*) > 1)) repeat_people from customers""").T.to_string())

h("PRODUCTS")
print(q("""select count(*) n, sum((product_category_name is null)::int) no_category, sum((product_weight_g is null)::int) no_weight,
  sum((product_weight_g = 0)::int) zero_weight, sum((product_photos_qty is null)::int) no_photos,
  (select count(*) from (select distinct product_category_name from products) anti join cat_tr using(product_category_name)) untranslated_categories
  from products""").T.to_string())

h("SELLERS")
print(q("""select (select count(*) from sellers) sellers, (select count(*) from sellers anti join items using(seller_id)) sellers_without_items,
  (select count(distinct seller_id) from deals) deal_sellers,
  (select count(*) from (select distinct seller_id from deals) d anti join sellers using(seller_id)) deal_sellers_not_in_seller_table,
  (select count(*) from (select distinct seller_id from deals) d semi join items using(seller_id)) deal_sellers_with_items""").T.to_string())

h("GEOLOCATION")
print(q("""select count(*) n_rows, count(distinct geolocation_zip_code_prefix) zips,
  sum((geolocation_lat not between -34 and 6 or geolocation_lng not between -74 and -34)::int) outside_brazil_box from geo""").T.to_string())
print(q("""select (select count(distinct customer_zip_code_prefix) from customers c where customer_zip_code_prefix not in (select geolocation_zip_code_prefix from geo)) customer_zips_missing,
  (select count(*) from customers where customer_zip_code_prefix not in (select geolocation_zip_code_prefix from geo)) customers_missing_geo,
  (select count(*) from sellers where seller_zip_code_prefix not in (select geolocation_zip_code_prefix from geo)) sellers_missing_geo""").T.to_string())

h("MQL and DEALS")
print(q("""select min(first_contact_date) first_mql, max(first_contact_date) last_mql, count(*) n, count(distinct landing_page_id) pages,
  sum((origin is null)::int) no_origin from mql""").T.to_string())
print(q("""select origin, count(*) n from mql group by 1 order by 2 desc""").to_string())
print(q("""select strftime(first_contact_date, '%Y-%m') m, count(*) mql from mql group by 1 order by 1""").T.to_string())
print(q("""select strftime(won_date, '%Y-%m') m, count(*) won from deals group by 1 order by 1""").T.to_string())
print(q("""select count(*) deals, sum((d.won_date < m.first_contact_date)::int) won_before_contact,
  median(epoch(d.won_date - m.first_contact_date)/86400) median_days_to_close,
  quantile_cont(epoch(d.won_date - m.first_contact_date)/86400, 0.9) p90_days_to_close,
  count(distinct sdr_id) sdrs, count(distinct sr_id) srs
  from deals d join mql m using(mql_id)""").T.to_string())
cols = ["business_segment", "lead_type", "lead_behaviour_profile", "has_company", "has_gtin", "average_stock", "business_type",
        "declared_product_catalog_size", "declared_monthly_revenue"]
print(q("select " + ", ".join(f"avg(({c} is null)::int) {c}" for c in cols) + " from deals").T.rename(columns={0: "null_share"}).to_string())
print(q("""select sum((declared_monthly_revenue = 0)::int) revenue_zero, sum((declared_monthly_revenue > 0)::int) revenue_positive from deals""").to_string())
print(q("""select sr_id, count(*) deals from deals group by 1 order by 2 desc""").to_string())
