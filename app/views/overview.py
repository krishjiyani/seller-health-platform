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

c = o[o.clean]
late = c[(c.late_delivery == 1) & c.late_dispatch.notna()]
late_rev = c[(c.late_delivery == 1) & c.ts_review.notna() & c.ts_delivered.notna()]
low = c[c.low_review == 1]
runway = d[d.won_date <= data.LAST_ASOF - pd.Timedelta(days=90)]
cols = st.columns(5)
ui.stat(cols[0], "Late deliveries not caused by the seller", ui.pct((late.late_dispatch == 0).mean(), 0),
        "Of parcels that reached the customer after the promised date or never arrived, the share where the seller had handed "
        "over to the carrier on time.")
ui.stat(cols[1], "Reviews on late orders written before arrival", ui.pct((late_rev.ts_review < late_rev.ts_delivered.dt.normalize()).mean(), 0),
        "The review survey goes out 2 days after the promised date if the parcel has not arrived (measured), so the bad review "
        "is locked in when the promise breaks.")
ui.stat(cols[2], "Bad reviews on orders that arrived on time", ui.pct((low.outcome == "On time").mean(), 0),
        "Share of all 1 or 2 star reviews. Lateness explains only part of the problem; see the Why Bad Reviews page.")
ui.stat(cols[3], "Unfair 1 or 2 star reviews", f"{int(c.unjust_review.sum()):,}",
        "Bad reviews on orders the seller dispatched on time but the carrier delivered late or lost.")
ui.stat(cols[4], "Signed sellers who never make a sale", ui.pct(runway.first_sale.isna().mean(), 0),
        "Deals won at least 90 days before the data ends, with no order by 31 Aug 2018.")
st.caption("Orders promised by 29 Aug 2018 (the public data stops collecting reviews on 31 Aug 2018). Late means delivered on a "
           "later day than promised, or never delivered.")

st.subheader("The four intervention windows")
def daily(k):
    df = pd.DataFrame(m[k])
    return df[df.month >= "2017-09"]
dd, cd = daily("dispatch_daily"), daily("carrier_daily")
sil = pd.DataFrame(m["silence"])
sil = sil[sil.month >= "2017-09"]
windows = pd.DataFrame([
    ["Dispatch window", "An order will miss the seller's dispatch deadline", "About 6 days (purchase to deadline)",
     f"AUC {dd.auc.mean():.2f} (best rule {dd.auc_days_waiting_only.mean():.2f}); High tier: {dd.high_rate.mean():.0%} late",
     "Call the seller to dispatch before the deadline"],
    ["Carrier window", "A parcel will arrive after the promised date, or never", "About 20 days (handover to survey date)",
     f"AUC {cd.auc.mean():.2f} (best rule {cd.auc_days_left_only.mean():.2f}); High tier: {cd.high_rate.mean():.0%} late",
     "Warn the customer and escalate to logistics before the survey goes out"],
    ["Onboarding window", "A lead or a newly signed seller will go cold", "Day 0 to 14 (leads), day 0 to 44 (signed sellers)",
     "Measured decay curves, no model", "Follow up the lead, or book a setup call, while the chance is still high"],
    ["Silence window", "An active seller will go silent within 1 to 3 months", "1 to 3 months",
     f"AUC {sil[sil.horizon == 1].auc.mean():.2f} / {sil[sil.horizon == 3].auc.mean():.2f}",
     "Check-in call, reason taken from the seller's warning signals"],
], columns=["Window", "What is predicted", "Time to act", "Accuracy on unseen months", "KAM action"])
st.dataframe(windows, hide_index=True, width="stretch")
st.caption("Order windows: risk re-scored daily and averaged over Sep 2017 to Aug 2018. AUC: the chance a real positive case is ranked "
           "above a real negative one; 0.50 is a coin flip, 0.70 is useful, 0.80 is strong. High tier: the riskiest 10% of each "
           "day's open orders, and the share of them that really went late.")

st.subheader("How to use this demo")
st.markdown(
    "1. Pick an **as-of date** in the sidebar. Every page shows only what was known on that day.\n"
    "2. Open the **KAM Console** to see this week's actions across the four windows.\n"
    "3. Turn on **Reveal what actually happened** to check the predictions against the real future.\n"
    "4. Open any seller in **Seller Drilldown** to see the orders and reviews behind every number.\n"
    "5. **Why Bad Reviews** shows what customers complain about when the parcel was on time.\n"
    "6. **Evidence and Limits** shows how every number was tested, what the data cannot predict, and how each label is defined."
)
