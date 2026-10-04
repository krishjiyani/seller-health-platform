"""Seller Health and Intervention Platform. Run from the repo root: streamlit run app/streamlit_app.py"""
import streamlit as st

st.set_page_config(page_title="Seller Health and Intervention Platform", page_icon="📦", layout="wide")

pages = {
    "Platform": [
        st.Page("views/overview.py", title="Overview", icon=":material/home:", default=True),
        st.Page("views/console.py", title="KAM Console", icon=":material/support_agent:"),
        st.Page("views/seller.py", title="Seller Drilldown", icon=":material/storefront:"),
    ],
    "Funnels": [
        st.Page("views/order_journey.py", title="Order Journey", icon=":material/local_shipping:"),
        st.Page("views/onboarding.py", title="Seller Onboarding", icon=":material/how_to_reg:"),
        st.Page("views/lifecycle.py", title="Seller Lifecycle", icon=":material/timeline:"),
    ],
    "Proof": [
        st.Page("views/evidence.py", title="Evidence and Limits", icon=":material/fact_check:"),
    ],
}
st.navigation(pages).run()
