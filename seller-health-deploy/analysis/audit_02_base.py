"""Builds order-level base table (one row per order) and saves parquet.
Attribution rule: order belongs to the seller with the highest item value in it (ties: lowest seller_id).
Review dedupe: keep the latest review_answer_timestamp per order."""
from common import connect
from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "data" / "interim"
con = connect()
con.execute("""
create table seller_val as
select order_id, seller_id, sum(price) v, sum(freight_value) f, count(*) n,
       max(shipping_limit_date) lim
from items group by 1,2;
create table main_seller as
select * exclude rn from (
  select order_id, seller_id, row_number() over(partition by order_id order by v desc, seller_id) rn
  from seller_val) where rn=1;
create table item_agg as
select i.order_id, sum(price) price, sum(freight_value) freight, count(*) n_items,
       count(distinct i.seller_id) n_sellers, max(shipping_limit_date) ship_limit,
       sum(coalesce(p.product_weight_g,0)) weight_g,
       arg_max(coalesce(t.product_category_name_english, p.product_category_name), price) category,
       avg(p.product_photos_qty) photos, avg(p.product_description_lenght) desc_len,
       arg_max(i.product_id, price) main_product
from items i left join products p using(product_id)
left join cat_tr t using(product_category_name) group by 1;
create table rev as
select * exclude rn from (
  select order_id, review_score, review_comment_title, review_comment_message, review_creation_date,
         review_answer_timestamp,
         row_number() over(partition by order_id order by review_answer_timestamp desc, review_id) rn
  from reviews) where rn=1;
create table pay as
select order_id, arg_max(payment_type, payment_value) pay_type, max(payment_installments) installments,
       sum(payment_value) pay_value from payments group by 1;
create table geo_c as select geolocation_zip_code_prefix zip, avg(geolocation_lat) lat, avg(geolocation_lng) lng
from geo group by 1;
create table base as
select o.order_id, o.order_status status, o.order_purchase_timestamp ts_purchase, o.order_approved_at ts_approved,
       o.order_delivered_carrier_date ts_carrier, o.order_delivered_customer_date ts_delivered,
       o.order_estimated_delivery_date ts_estimated,
       m.seller_id, a.* exclude(order_id), c.customer_unique_id, c.customer_state, s.seller_state,
       c.customer_zip_code_prefix cust_zip, s.seller_zip_code_prefix seller_zip,
       r.review_score, r.review_comment_message, r.review_comment_title, r.review_creation_date,
       p.pay_type, p.installments, p.pay_value,
       2*6371*asin(sqrt(pow(sin(radians(gc.lat-gs.lat)/2),2)+cos(radians(gs.lat))*cos(radians(gc.lat))*pow(sin(radians(gc.lng-gs.lng)/2),2))) dist_km
from orders o
left join main_seller m using(order_id)
left join item_agg a using(order_id)
left join customers c using(customer_id)
left join sellers s on s.seller_id=m.seller_id
left join rev r using(order_id)
left join pay p using(order_id)
left join geo_c gc on gc.zip=c.customer_zip_code_prefix
left join geo_c gs on gs.zip=s.seller_zip_code_prefix
""")
n = con.execute("select count(*), count(distinct order_id) from base").fetchone()
assert n[0] == n[1] == 99441, n
print("base rows", n, "no seller (no items):", con.execute("select count(*) from base where seller_id is null").fetchone())
print("dist null share", con.execute("select avg((dist_km is null)::int) from base").fetchone())
con.execute(f"copy base to '{OUT/'orders_base.parquet'}' (format parquet)")
con.execute(f"copy (select d.*, m.first_contact_date, m.landing_page_id, m.origin from deals d join mql m using(mql_id)) to '{OUT/'deals.parquet'}' (format parquet)")
con.execute(f"copy mql to '{OUT/'mql.parquet'}' (format parquet)")
print("saved")
