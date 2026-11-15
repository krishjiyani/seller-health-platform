import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import data, ui

T, reveal = ui.sidebar_controls()
m = data.metrics()
o = data.orders()
st.title("Evidence and Limits")
st.markdown("How every prediction was tested, how well it held up, and what the data **cannot** support. "
            "Each month was scored by a model trained only on outcomes known before that month started (walk forward validation). "
            "Training and testing use orders promised by 29 Aug 2018, the last date the public data can judge.")

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
def row(name, df, rule_col, rule_name):
    df = df[df.month >= FROM]
    beat = int((df.auc > df[rule_col]).sum())
    return [name, f"{df.auc.mean():.3f} ± {df.auc.std():.3f}", f"{df[rule_col].mean():.3f} ({rule_name})", f"{beat} of {len(df)}",
            ui.pct(df.base_rate.mean()), ui.pct(df.top10_rate.mean()), f"{df.top10_rate.mean() / df.base_rate.mean():.1f}x"]
summary = pd.DataFrame(
    [row("Dispatch window", dd, "auc_days_waiting_only", "days waiting"),
     row("Carrier window", cd, "auc_days_left_only", "days left")] +
    [row(f"Silence within {k} month(s)", sil[sil.horizon == k], "auc_recency_only", "recency") for k in (1, 2, 3)],
    columns=["Prediction", "Model AUC (mean ± sd)", "Best simple rule AUC", "Months the model beat the rule", "Base rate",
             "Top 10% hit rate", "Lift"])
st.dataframe(summary, hide_index=True, width="stretch")
st.markdown(
    "**Reading the table.** Dispatch: the model clearly beats the best rule in every month; it reads the seller's dispatch habit and "
    "what the seller is doing right now (open backlog, late handovers in the last 14 days, a surge in orders). Carrier: the gain "
    "over \"days left before the promise\" is real but small, so the honest claim is that timing drives this window and the model "
    "adds the cause (seller or carrier) and the lane. Silence: a refined recency signal. Lift: how much more often the riskiest 10% "
    "turn out positive than an average case.")

st.subheader("What a KAM gets from the High tier")
st.markdown("The console shows tiers instead of percentages: **High** is the riskiest 10% of each day's open orders, **Elevated** the "
            "next 20%. Measured on every test month from Sep 2017 to Aug 2018.")


def tier_row(name, daily_df, orders_df):
    daily_df = daily_df[daily_df.month >= FROM]
    od = orders_df[orders_df.month >= FROM]
    recall = (od.recall * od.late).sum() / od.late.sum()
    precision = (od.precision * od.flagged).sum() / od.flagged.sum()
    return [name, ui.pct(daily_df.high_rate.mean(), 0), ui.pct(daily_df.elevated_rate.mean(), 0), ui.pct(daily_df.normal_rate.mean(), 0),
            ui.pct(recall, 0), ui.pct(precision, 0), ui.pct(od.flagged.sum() / od.orders.sum(), 0),
            f"{od.warning_days_median.median():.1f} days"]
tiers = pd.DataFrame(
    [tier_row("Dispatch window (late or never dispatched)", dd, pd.DataFrame(m["dispatch_orders"])),
     tier_row("Carrier window (late or never delivered)", cd, pd.DataFrame(m["carrier_orders"]))],
    columns=["Window", "Late share: High", "Elevated", "Normal", "Late orders that reached High before the deadline",
             "High orders that went late", "Orders that reached High", "Median warning before the deadline"])
st.dataframe(tiers, hide_index=True, width="stretch")
st.caption("A KAM who works the High tier every day touches the share of orders in the seventh column and reaches the share of late "
           "orders in the fifth, with the warning in the last column. Percentages are not shown in the console because order "
           "probabilities drift with the promise policy and with carrier shocks, while the ranking holds.")

st.subheader("Seller silence by seller size")
bs = pd.DataFrame(m["silence_by_size"])
bs = bs.assign(horizon=bs.horizon.map({1: "1 month", 2: "2 months", 3: "3 months"}))
st.dataframe(bs.rename(columns={"horizon": "Horizon", "size": "Sellers", "n": "Seller months", "silent": "Went silent",
                                "auc_model": "Model AUC", "auc_recency": "Days since last order AUC", "auc_shipped": "Console AUC"}),
             hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.3f")
                                                              for c in ["Model AUC", "Days since last order AUC", "Console AUC"]})
st.caption("Pooled over Sep 2017 to Aug 2018. For sellers with 10 or more orders in 90 days, days since last order ranks better than the "
           "model, so the console scores them by that one signal (fitted on earlier months only). Everyone else keeps the model.")

st.subheader("How the labels are defined, and why")
st.markdown("Every label was checked against the raw data before any model was trained.")
delivered = o[o.ts_delivered.notna()]
reviewed = delivered[delivered.ts_review.notna()]
same_day = delivered[delivered.ts_delivered.dt.normalize() == delivered.ts_estimated]
on_time = delivered[delivered.late_delivery == 0]
lead = data.mql()
early_leads = lead[~lead.lead_cohort]
rules = pd.DataFrame([
    ["Late delivery: delivered on a later calendar day than the promised date",
     "The promised date has no time of day",
     f"Parcels delivered on the promised day get 1 or 2 stars {same_day.low_review.mean():.0%} of the time, like on time orders "
     f"({on_time.low_review.mean():.0%}), so they count as on time"],
    ["Parcels never delivered count as late deliveries; orders never handed over count as late dispatches",
     "These are the worst outcomes, and every deadline passed before the data ends",
     f"{int(o.lost.sum()):,} lost parcels ({o[o.lost].low_review.mean():.0%} bad reviews) and {int(o.never_dispatched.sum()):,} "
     f"orders never dispatched ({o[o.never_dispatched].low_review.mean():.0%} bad reviews)"],
    ["The carrier window ends on the review survey date",
     "The survey goes out the day after delivery, or 2 days after the promised date if the parcel has not arrived",
     f"Exact for {(reviewed.ts_review.dt.normalize() == reviewed.survey_due).mean():.1%} of delivered, reviewed orders"],
    ["Models train and are tested on orders promised by 29 Aug 2018",
     "Reviews stop on 31 Aug 2018 and the public data only keeps reviewed orders, so later orders are a fast delivery only sample",
     "Late August orders are shown in the console with a partial data note"],
    ["Lead numbers use leads first contacted from Dec 2017",
     "The deals file only holds wins from Dec 2017",
     f"Earlier leads win {early_leads.won.mean():.1%} of the time against {lead[lead.lead_cohort].won.mean():.1%}, an artifact of the file"],
], columns=["Rule", "Why", "Evidence"])
st.dataframe(rules, hide_index=True, width="stretch")

with st.expander("Why risk is re-scored daily"):
    st.markdown(
        f"Scoring each order once, at the start of its window, gives AUC {dis0[dis0.month >= FROM].auc.mean():.2f} for dispatch and "
        f"{car0[car0.month >= FROM].auc.mean():.2f} for carrier over all orders. But in a live queue, the orders still waiting are "
        "exactly the slow ones, and a score fixed at the start ranks them worse than \"days already waiting\" alone. A window model "
        "must update as the window closes, so each order is re-scored every day it stays open. Training also avoids a survivor bias: "
        "using only orders whose outcome was already known would keep the fast ones and drop the slow ones, so training uses orders "
        "that started their window at least 30 days (dispatch) or 60 days (carrier) before the scored month.")

st.subheader("What the data cannot predict")
st.dataframe(pd.DataFrame([
    ["Decline of established sellers (10+ orders in 90 days)", "AUC 0.61, below days since last order (0.64)", "The reason a seller declines is not recorded"],
    ["Abrupt seller exits", "62% of exits show no prior decline", "Nothing in the data changes before they stop"],
    ["Activation from signup details", "AUC 0.61", "Signup fields are thin; declared revenue is 0 for 797 of 842 deals"],
    ["Which leads will sign", "AUC 0.61 on leads from Dec 2017", "Only the channel and landing page are recorded; timing works better"],
    ["Cancellations at purchase", "AUC 0.61", "Too rare (0.4% of orders with items)"],
    ["Bad review at purchase", "AUC 0.63", "It depends on what happens after purchase; the order shape rule carries most of the signal"],
    ["Customer buying again", "AUC 0.57", "97% of customers buy once"],
    ["Effect of public holidays", "Late dispatch 9.2% vs 9.0%; no AUC gain", "No measurable effect"],
], columns=["Target", "Result", "Why"]), hide_index=True, width="stretch")

st.subheader("Hypotheses tested against natural shocks")
st.markdown(
    "* **Black Friday stress test (supported, small sample).** Of 179 sellers busy before and during Black Friday 2017, the 37 whose dispatch "
    "broke under load grew 0.93x over the next quarter versus 1.11x for those that held; 8% went silent versus 0%.\n"
    "* **Unfair reviews hurt sales (not supported, tested twice).** During the May 2018 truckers' strike, sellers who dispatched on time but "
    "whose parcels still arrived late got more bad reviews (14% vs 9%), yet their sales did not fall over the next two months (1.03x vs 0.88x). "
    "A wider test over all months compared weeks where a carrier caused delay drew a 1 or 2 star review with weeks where it did not "
    "(1,829 seller weeks): orders over the next 4 weeks moved by -5.5%, with a 95% interval of -12% to +2%, so no effect could be "
    "shown. The supported claim is **unfair reviews**, not unfair seller deaths.")

st.subheader("Rules and limitations")
st.markdown(
    "* Orders with several sellers (1.3%) are credited to the seller with the highest item value. A seller's first sale counts any order with their item.\n"
    "* Duplicate reviews: one per order, the latest answered.\n"
    "* Late means delivered on a later day than promised, or never delivered. Late dispatch includes orders never handed over.\n"
    "* Cancelled orders are left out of live queues because the data does not record when they were cancelled.\n"
    "* The onboarding, lead and new seller ramp curves are measured on the whole period, so early as-of dates see curves partly built from later data.\n"
    "* Complaint tags come from keyword rules checked by reading 200 reviews; they are precise but miss some complaints.\n"
    "* The data contains no interventions. Nothing here proves that acting helps; it shows when acting is possible.\n"
    "* Smaller data quirks, none material: 166 handovers logged before purchase (Apr to Aug 2018), 547 orders with several reviews, "
    "610 products without a category, one deal won 2 days before its first contact.")
