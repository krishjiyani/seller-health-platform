"""Shared loader: registers every raw CSV as a DuckDB view."""
from pathlib import Path
import duckdb

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
TABLES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geo": "olist_geolocation_dataset.csv",
    "cat_tr": "product_category_name_translation.csv",
    "mql": "olist_marketing_qualified_leads_dataset.csv",
    "deals": "olist_closed_deals_dataset.csv",
}

def connect():
    con = duckdb.connect()
    for name, f in TABLES.items():
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_csv_auto('{RAW / f}', header=true)")
    return con
