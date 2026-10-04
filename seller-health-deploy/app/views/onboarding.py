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
dd = d if lead == "All" else d[d.lead_type == lead]
stages = ([("Marketing qualified leads", len(q))] if lead == "All" else []) + [
    ("Deal won", len(dd)), ("First sale", int(dd.first_sale.notna().sum())),
    ("Still selling the month after", int((dd.active_month_after_first == 1).sum()))]
st.plotly_chart(ui.funnel(stages), width="stretch")
st.caption("Deals were won between Dec 2017 and Nov 2018 while orders end in Aug 2018, so late deals had little time to sell. "
           "The comparisons below use only deals won at least 90 days before the data ends.")

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
