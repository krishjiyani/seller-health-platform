from common import connect
con = connect()
q = lambda s: con.execute(s).fetchall()
expected = dict(orders=99441, items=112650, reviews=99224, payments=103886, customers=99441,
                products=32951, sellers=3095, geo=1000163, cat_tr=71, mql=8000, deals=842)
for t, e in expected.items():
    n = q(f"select count(*) from {t}")[0][0]
    print(f"{t:10s} {n:>9,} expected {e:>9,} {'OK' if n==e else 'MISMATCH'}")
print("orphans items->orders", q("select count(*) from items i anti join orders o using(order_id)")[0][0])
print("orphans items->sellers", q("select count(*) from items i anti join sellers s using(seller_id)")[0][0])
print("orphans deals->mql", q("select count(*) from deals anti join mql using(mql_id)")[0][0])
print("deal sellers that ever sold", q("select count(distinct d.seller_id) from deals d join items i using(seller_id)")[0][0])
print("status mix", q("select order_status, count(*) from orders group by 1 order by 2 desc"))
print("monthly orders", q("select strftime(order_purchase_timestamp,'%Y-%m') m, count(*) from orders group by 1 order by 1"))
# delivery decomposition
r = q("""
with lim as (select order_id, max(shipping_limit_date) lim from items group by 1)
select avg((o.order_delivered_carrier_date > l.lim)::int),
       avg((o.order_delivered_customer_date > o.order_estimated_delivery_date)::int),
       sum(((o.order_delivered_customer_date > o.order_estimated_delivery_date) and (o.order_delivered_carrier_date > l.lim))::int)
         / sum((o.order_delivered_customer_date > o.order_estimated_delivery_date)::int)
from orders o join lim l using(order_id)
where o.order_status='delivered' and o.order_delivered_customer_date is not null and o.order_delivered_carrier_date is not null""")
print("late handover, late delivery, share of late with late handover", r)
print("won_date range", q("select min(won_date), max(won_date) from deals"))
