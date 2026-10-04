import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import data, ui

T, reveal = ui.sidebar_controls()
o = data.orders()
sm = data.seller_months()
st.title("Seller Drilldown")

counts = o[o.seller_id.notna()].groupby("seller_id").size().sort_values(ascending=False)
st.session_state.setdefault("seller", counts.index[0])
options = list(counts.index)
sid = st.selectbox("Seller", options, index=options.index(st.session_state.seller) if st.session_state.seller in options else 0,
                   format_func=lambda s: f"{s[:8]}  ({counts[s]:,} orders)")
st.session_state.seller = sid

so = o[(o.seller_id == sid) & (o.ts_purchase <= T)].sort_values("ts_purchase")
h = sm[(sm.seller_id == sid) & (sm.month <= data.panel_month_for(T))].sort_values("month")
if so.empty:
    st.warning(f"This seller had no orders by {T:%d %b %Y}. Move the as-of date later in the sidebar.")
    st.stop()
last = h.iloc[-1] if len(h) else None
st.caption(f"Everything below uses data up to **{T:%d %b %Y}** only. Seller state: {so.seller_state.iloc[-1]}. "
           f"Main category: {so.category.mode().iloc[0]}.")

c = st.columns(5)
ui.stat(c[0], "Orders to date", f"{len(so):,}")
ui.stat(c[1], "GMV, last 90 days", f"R$ {last.g_90:,.0f}" if last is not None else "n/a")
ui.stat(c[2], "Late dispatch, last 180 days", ui.pct(last.late_dispatch_180) if last is not None else "n/a",
        "The seller's own share of orders handed to the carrier after the deadline. Platform average is about 9%.")
ui.stat(c[3], "Review score (adjusted)", f"{last.review_shrunk_180:.2f}" if last is not None else "n/a",
        "Average stars over 180 days, pulled toward the platform average when the seller has few reviews.")
ui.stat(c[4], "Unfair bad reviews", f"{int(so.unjust_review.sum())}",
        "1 or 2 star reviews on orders this seller dispatched on time but the carrier delivered late.")

if last is not None and pd.notna(last.p_silent_1):
    sig = data.signals(last)
    st.markdown(f"**Silence forecast ({last.month}):** next month {ui.pct(last.p_silent_1, 0)}, within 3 months {ui.pct(last.p_silent_3, 0)}. "
                + ("Warning signals: " + "; ".join(sig) if sig else "No warning signals."))

left, right = st.columns(2)
with left:
    st.markdown("**Orders per month**")
    m = so.groupby(so.ts_purchase.dt.to_period("M")).size()
    m.index = m.index.to_timestamp()
    fig = go.Figure(go.Bar(x=m.index, y=m.values, marker=dict(color=ui.BLUE, cornerradius=4),
                           hovertemplate="%{x|%b %Y}: %{y} orders<extra></extra>"))
    st.plotly_chart(ui.style(fig, 280, legend=False), width="stretch")
with right:
    st.markdown("**Silence forecast over time**")
    hh = h[h.p_silent_1.notna()]
    fig = go.Figure()
    x = pd.PeriodIndex(hh.month, freq="M").to_timestamp(how="end")
    fig.add_scatter(x=x, y=hh.p_silent_1, name="Silent next month", line=dict(color=ui.BLUE, width=2), mode="lines+markers", marker=dict(size=8))
    fig.add_scatter(x=x, y=hh.p_silent_3, name="Silent within 3 months", line=dict(color=ui.ORANGE, width=2), mode="lines+markers", marker=dict(size=8))
    fig.add_hline(y=data.HIGH_SILENCE, line=dict(color=ui.MUTED, width=1, dash="dot"))
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.update_traces(hovertemplate="%{x|%b %Y}: %{y:.0%}")
    st.plotly_chart(ui.style(fig, 280), width="stretch")

st.markdown("**Evidence: this seller's orders** (newest first). Review text is the customer's own words, in Portuguese.")
ev = so.sort_values("ts_purchase", ascending=False).head(300).copy()
ev["dispatch"] = ev.late_dispatch.map({1.0: "Late", 0.0: "On time"}).fillna("Pending or none")
ev["delivery"] = ev.late_delivery.map({1.0: "Late", 0.0: "On time"}).fillna("Pending or none")
ev.loc[ev.ts_carrier > T, "dispatch"] = "Pending"
ev.loc[ev.ts_delivered > T, "delivery"] = "Pending"
ev.loc[ev.ts_review > T, ["review_score", "review_text"]] = None
ev["unfair"] = ev.unjust_review & (ev.ts_review <= T)
st.dataframe(ev[["order_id", "ts_purchase", "category", "price", "dispatch", "delivery", "review_score", "unfair", "review_text"]]
             .assign(order_id=ev.order_id.str[:8]), hide_index=True, width="stretch", column_config={
    "order_id": "Order", "ts_purchase": st.column_config.DatetimeColumn("Bought", format="DD MMM YYYY"),
    "category": "Category", "price": st.column_config.NumberColumn("Price (R$)", format="%.2f"),
    "dispatch": "Dispatch", "delivery": "Delivery", "review_score": "Stars",
    "unfair": st.column_config.CheckboxColumn("Unfair review"), "review_text": st.column_config.TextColumn("Review", width="large")})
