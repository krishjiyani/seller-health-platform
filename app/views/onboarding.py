import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import data, ui

T, reveal = ui.sidebar_controls()
d = data.deals()
q = data.mql()
st.title("Seller Onboarding Funnel")
st.markdown("From marketing lead to a seller who keeps selling. The biggest leak in the business is **after** the deal is signed: "
            "many signed sellers never list a product that sells.")

lead = st.selectbox("Lead type", ["All"] + sorted(d.lead_type.dropna().unique()))
cohort = q[q.lead_cohort]
dd = d if lead == "All" else d[d.lead_type == lead]
if lead == "All":
    dd = dd[dd.mql_id.isin(cohort.mql_id)]
stages = ([("Marketing qualified leads (Dec 2017 to May 2018)", len(cohort))] if lead == "All" else []) + [
    ("Deal won", len(dd)), ("First sale", int(dd.first_sale.notna().sum())),
    ("Still selling the month after", int((dd.active_month_after_first == 1).sum()))]
st.plotly_chart(ui.funnel(stages), width="stretch")
st.caption("The deals file only holds wins from Dec 2017, so leads first contacted earlier look like losers (2.8% win rate against "
           "12.8%). The lead stage therefore uses leads first contacted from Dec 2017. Deals were won up to Nov 2018 while orders end "
           "in Aug 2018, so late deals had little time to sell; the comparisons below use only deals won at least 90 days before the data ends.")

st.subheader("The lead window: follow up within 14 days")
decay = data.lead_decay()
fig = go.Figure(go.Scatter(x=decay.day, y=decay.p_win, mode="lines+markers", line=dict(color=ui.BLUE, width=2), marker=dict(size=8),
                           customdata=decay.open_leads,
                           hovertemplate="Not signed by day %{x}: %{y:.1%} still sign (n=%{customdata:,})<extra></extra>"))
fig.update_yaxes(tickformat=".0%", range=[0, max(0.16, decay.p_win.max() + 0.02)], title="Chance the lead still signs")
fig.update_xaxes(title="Days since first contact, still not signed")
st.plotly_chart(ui.style(fig, 320, legend=False), width="stretch")
dz = decay.set_index("day").p_win
wins = cohort[cohort.won]
close_median = ((wins.won_date - wins.first_contact_date).dt.total_seconds() / 86400).clip(lower=0).median()
st.markdown(f"A new lead signs **{dz.loc[0]:.1%}** of the time. One still unsigned after 7 days signs **{dz.loc[7]:.1%}** of the time, "
            f"after 14 days **{dz.loc[14]:.1%}**, after 30 days **{dz.loc[30]:.1%}**. Half of all wins close within {close_median:.0f} days. "
            "Which leads sign could not be predicted from the lead's channel or landing page, so timing sets the follow up.")

runway = d[d.won_date <= data.LAST_ASOF - pd.Timedelta(days=90)].assign(activated=lambda x: x.first_sale.notna())
left, right = st.columns(2)
with left:
    st.markdown("**Activation rate by lead type**")
    g = runway.groupby("lead_type").activated.agg(["mean", "size"]).sort_values("mean")
    fig = go.Figure(go.Bar(x=g["mean"], y=g.index, orientation="h", marker=dict(color=ui.BLUE, cornerradius=4),
                           text=[f"{v:.0%}  (n={n})" for v, n in zip(g["mean"], g["size"])], textposition="outside",
                           cliponaxis=False, hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(ui.style(fig, 300, legend=False), width="stretch")
with right:
    st.markdown("**Activation rate by business type**")
    g = runway.groupby("business_type").activated.agg(["mean", "size"]).sort_values("mean")
    fig = go.Figure(go.Bar(x=g["mean"], y=g.index, orientation="h", marker=dict(color=ui.BLUE, cornerradius=4),
                           text=[f"{v:.0%}  (n={n})" for v, n in zip(g["mean"], g["size"])], textposition="outside",
                           cliponaxis=False, hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(ui.style(fig, 300, legend=False), width="stretch")
st.markdown("Sellers already selling online activate far more often than offline businesses, while the marketing channel barely matters. "
            "This is an **onboarding friction** problem, not a marketing problem.")

st.subheader("The onboarding window: every week of waiting costs activation")
curve = data.onboarding_curve()
fig = go.Figure(go.Scatter(x=curve.day, y=curve.p_sell_by_90, mode="lines+markers", line=dict(color=ui.BLUE, width=2), marker=dict(size=8),
                           customdata=curve.still_unsold, hovertemplate="Unsold at day %{x}: %{y:.0%} will still sell by day 90 (n=%{customdata})<extra></extra>"))
fig.update_yaxes(tickformat=".0%", range=[0, max(0.6, curve.p_sell_by_90.max() + 0.05)], title="Chance of first sale by day 90")
fig.update_xaxes(title="Days since deal won, still without a sale")
st.plotly_chart(ui.style(fig, 340, legend=False), width="stretch")
lag = d.days_to_first_sale.dropna()
st.caption(f"Among sellers who did activate: median {lag.median():.0f} days from deal to first sale, 90th percentile {lag.quantile(.9):.0f} days. "
           "A signup attribute model scored AUC 0.61, so urgency is set by elapsed days, not by a prediction.")
