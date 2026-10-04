import pandas as pd
import streamlit as st
from lib import data, ui

T, reveal = ui.sidebar_controls()
o = data.orders()
m = data.metrics()
d = data.deals()

st.title("Seller Health and Intervention Platform")
st.markdown(
    "A decision support console for **key account managers (KAMs)** on an ecommerce marketplace. "
    "It predicts **when** to step in, at order level and seller level, while there is still time to prevent the damage, "
    "and recommends what to do. Every prediction was tested on months the model had never seen."
)

late = o[(o.late_delivery == 1) & o.late_dispatch.notna()]
late_rev = o[(o.late_delivery == 1) & o.ts_review.notna() & o.ts_delivered.notna()]
runway = d[d.won_date <= data.LAST_ASOF - pd.Timedelta(days=90)]
c1, c2, c3, c4 = st.columns(4)
ui.stat(c1, "Late deliveries not caused by the seller", ui.pct((late.late_dispatch == 0).mean()),
        "Of orders that reached the customer late, the share where the seller had handed over to the carrier on time.")
ui.stat(c2, "Reviews on late orders written before arrival", ui.pct((late_rev.ts_review < late_rev.ts_delivered.dt.normalize()).mean(), 0),
        "The review survey appears to fire near the promised date, so the bad review is locked in when the promise breaks.")
ui.stat(c3, "Signed sellers who never make a sale", ui.pct(runway.first_sale.isna().mean(), 0),
        "Deals won at least 90 days before the data ends, with no order by 31 Aug 2018.")
ui.stat(c4, "Unfair 1 or 2 star reviews", f"{int(o.unjust_review.sum()):,}",
        "Bad reviews on orders the seller dispatched on time but the carrier delivered late.")

st.subheader("The four intervention windows")
def mean(k):
    df = pd.DataFrame(m[k])
    return df[df.month >= "2017-09"].auc.mean()
sil = pd.DataFrame(m["silence"])
windows = pd.DataFrame([
    ["Dispatch window", "An order will miss the seller's dispatch deadline", "About 6 days (purchase to deadline)",
     f"AUC {mean('dispatch_daily'):.2f}", "Call the seller to dispatch before the deadline"],
    ["Carrier window", "A parcel with the carrier will arrive after the promised date", "About 20 days (handover to promise)",
     f"AUC {mean('carrier_daily'):.2f}", "Escalate to logistics and warn the customer before the promise breaks"],
    ["Onboarding window", "A newly signed seller will never start selling", "Day 0 to 44 after signing",
     "Measured decay curve, no model", "Setup call while the chance of activating is still high"],
    ["Silence window", "An active seller will go silent within 1 to 3 months", "1 to 3 months",
     f"AUC {sil[sil.horizon == 1].auc.mean():.2f} / {sil[sil.horizon == 3].auc.mean():.2f}", "Check-in call, reason taken from the seller's warning signals"],
], columns=["Window", "What is predicted", "Time to act", "Accuracy on unseen months", "KAM action"])
st.dataframe(windows, hide_index=True, width="stretch")
st.caption("Order windows: risk re-scored daily, averaged over Sep 2017 to Aug 2018. AUC: the chance a real positive case is ranked above a real negative one. 0.50 is a coin flip; 0.70 is useful; 0.80 is strong.")

st.subheader("How to use this demo")
st.markdown(
    "1. Pick an **as-of date** in the sidebar. Every page shows only what was known on that day.\n"
    "2. Open the **KAM Console** to see this week's actions across the four windows.\n"
    "3. Turn on **Reveal what actually happened** to check the predictions against the real future.\n"
    "4. Open any seller in **Seller Drilldown** to see the orders and reviews behind every number.\n"
    "5. The **Evidence and Limits** page shows what the data can and cannot predict, including the hypotheses that failed."
)
