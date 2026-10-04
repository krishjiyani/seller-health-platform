import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import data, ui

T, reveal = ui.sidebar_controls()
m = data.metrics()
st.title("Evidence and Limits")
st.markdown("How every prediction was tested, how well it held up, and what the data **cannot** support. "
            "Each month was scored by a model trained only on outcomes known before that month started (walk forward validation).")

dd, cd, sil = pd.DataFrame(m["dispatch_daily"]), pd.DataFrame(m["carrier_daily"]), pd.DataFrame(m["silence"])
dis0, car0 = pd.DataFrame(m["dispatch"]), pd.DataFrame(m["carrier"])
FROM = "2017-09"


def trend(df, model_name, rule_col, rule_name):
    fig = go.Figure()
    fig.add_scatter(x=df.month, y=df.auc, name=model_name, mode="lines+markers", line=dict(color=ui.BLUE, width=2), marker=dict(size=8))
    fig.add_scatter(x=df.month, y=df[rule_col], name=rule_name, mode="lines+markers",
                    line=dict(color=ui.MUTED, width=2, dash="dash"), marker=dict(size=8))
    fig.add_vrect(x0=df.month.min(), x1="2017-08", fillcolor=ui.GRID, opacity=0.4, line_width=0)
    fig.update_traces(hovertemplate="%{x}: AUC %{y:.3f}")
    fig.update_yaxes(range=[0.5, 0.95], title="AUC")
    return ui.style(fig, 300)


st.subheader("Accuracy month by month, against the best simple rule")
st.caption("Shaded months had under six months of training history. Headline numbers use September 2017 onward.")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Dispatch window** (risk re-scored daily)")
    st.plotly_chart(trend(dd, "Model", "auc_days_waiting_only", "Days already waiting"), width="stretch")
with c2:
    st.markdown("**Carrier window** (risk re-scored daily)")
    st.plotly_chart(trend(cd, "Model", "auc_days_left_only", "Days left before promise"), width="stretch")
st.markdown("**Seller silence** (next month)")
s1 = sil[sil.horizon == 1]
st.plotly_chart(trend(s1, "Model", "auc_recency_only", "Days since last order"), width="stretch")

st.subheader("Summary")
def row(name, df, window, rule_col, rule_name):
    df = df[df.month >= FROM]
    return [name, window, f"{df.auc.mean():.3f} ± {df.auc.std():.3f}", f"{df[rule_col].mean():.3f} ({rule_name})",
            ui.pct(df.base_rate.mean()), ui.pct(df.top10_rate.mean()), f"{df.top10_rate.mean() / df.base_rate.mean():.1f}x", len(df)]
summary = pd.DataFrame(
    [row("Dispatch window", dd, "About 6 days", "auc_days_waiting_only", "days waiting"),
     row("Carrier window", cd, "About 20 days", "auc_days_left_only", "days left")] +
    [row(f"Silence within {k} month(s)", sil[sil.horizon == k], f"{k} month(s)", "auc_recency_only", "recency") for k in (1, 2, 3)],
    columns=["Prediction", "Time to act", "Model AUC (mean ± sd)", "Best simple rule AUC", "Base rate", "Top 10% hit rate", "Lift", "Months"])
st.dataframe(summary, hide_index=True, width="stretch")
st.markdown(
    "**Reading the table.** Dispatch: the model clearly beats the best rule; the main driver is the seller's own dispatch habit. "
    "Carrier: the gain over \"days left before the promise\" is small, so the honest claim is that timing drives this window and the model "
    "adds the cause (seller or carrier) and the lane. The carrier feature set was chosen on these same months, so its true gain is likely a "
    "little smaller. Silence: a refined recency signal.")
with st.expander("Why risk is re-scored daily (a mistake we caught)"):
    st.markdown(
        f"The first version scored each order once, at the start of its window (dispatch AUC {dis0[dis0.month >= FROM].auc.mean():.2f}, "
        f"carrier {car0[car0.month >= FROM].auc.mean():.2f}, measured over all orders). But in a live queue, the orders still waiting are "
        "exactly the slow ones, and that version ranked them worse than \"days already waiting\" alone. A window model must update as the "
        "window closes, so each order is now re-scored every day it stays open. A second bias was also removed: training only on orders "
        "whose outcome was already known kept the fast ones and dropped the slow ones, so training now uses orders that started their "
        "window at least 30 days (dispatch) or 60 days (carrier) earlier.")

st.subheader("What the data cannot predict")
st.dataframe(pd.DataFrame([
    ["Decline of established sellers (10+ orders in 90 days)", "AUC 0.61, below days since last order (0.64)", "The reason a seller declines is not recorded"],
    ["Abrupt seller exits", "62% of exits show no prior decline", "Nothing in the data changes before they stop"],
    ["Activation from signup details", "AUC 0.61", "Signup fields are thin; declared revenue is 0 for 797 of 842 deals"],
    ["Cancellations at purchase", "AUC 0.61", "Too rare (0.4% of orders with items)"],
    ["Bad review at purchase", "AUC 0.63", "It depends on what happens after purchase"],
    ["Customer buying again", "AUC 0.57", "97% of customers buy once"],
    ["Effect of public holidays", "Late dispatch 9.2% vs 9.0%; no AUC gain", "No measurable effect"],
], columns=["Target", "Result", "Why"]), hide_index=True, width="stretch")

st.subheader("Hypotheses tested against natural shocks")
st.markdown(
    "* **Black Friday stress test (supported, small sample).** Of 179 sellers busy before and during Black Friday 2017, the 37 whose dispatch "
    "broke under load grew 0.93x over the next quarter versus 1.11x for those that held; 8% went silent versus 0%.\n"
    "* **Unfair reviews hurt sales (not supported).** During the May 2018 truckers' strike, sellers who dispatched on time but whose parcels "
    "still arrived late got more bad reviews (14% vs 9%), yet their sales did not fall over the next two months (1.03x vs 0.88x). "
    "The supported claim is **unfair reviews**, not unfair seller deaths.")

st.subheader("Rules and limitations")
st.markdown(
    "* Orders with several sellers (1.3%) are credited to the seller with the highest item value. A seller's first sale counts any order with their item.\n"
    "* Duplicate reviews: one per order, the latest answered.\n"
    "* Cancelled orders are left out of live queues because the data does not record when they were cancelled.\n"
    "* The onboarding decay curve uses all deals with 90 days of runway, so early as-of dates see a curve partly built from later deals.\n"
    "* The data contains no interventions. Nothing here proves that acting helps; it shows when acting is possible.\n"
    "* Review timing is inferred from timestamps, not from Olist documentation.\n"
    "* The last month (Aug 2018) is partly censored: many of its orders were not delivered when the data ends.")
