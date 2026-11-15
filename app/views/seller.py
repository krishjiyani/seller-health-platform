import numpy as np
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
reviewed = so[so.ts_review <= T]
ui.stat(c[4], "Unfair bad reviews", f"{int(reviewed.unjust_review.sum())}",
        "1 or 2 star reviews on orders this seller dispatched on time but the carrier delivered late or lost.")

if last is not None and pd.notna(last.p_silent_1):
    sig = data.signals(last)
    basis = " Scored by days since last order (10+ orders in 90 days)." if pd.notna(last.get("big_seller")) and bool(last.get("big_seller")) else ""
    st.markdown(f"**Silence forecast ({last.month}):** next month {ui.pct(last.p_silent_1, 0)}, within 3 months {ui.pct(last.p_silent_3, 0)}.{basis} "
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

bad = reviewed[reviewed.low_review == 1]
st.markdown(f"**Why this seller's bad reviews happened** ({len(bad):,} reviews with 1 or 2 stars by this date)")
if len(bad):
    causes = bad.bad_review_cause.value_counts()
    left, right = st.columns([3, 2])
    fig = go.Figure(go.Bar(x=causes.values, y=causes.index, orientation="h", marker=dict(color=ui.BLUE, cornerradius=4),
                           text=[f"{v}  ({v / len(bad):.0%})" for v in causes.values], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(visible=False, range=[0, causes.max() * 1.5])
    left.plotly_chart(ui.style(fig, 60 + 40 * len(causes), legend=False), width="stretch")
    seller_side = bad.bad_review_cause.isin(["Seller dispatched late or never", "Wrong, defective or missing item (from text)"]).mean()
    right.markdown(f"**{seller_side:.0%}** of these point to something the seller controls (late dispatch, or a wrong, defective or "
                   f"missing item); **{(bad.bad_review_cause == 'Carrier late or lost (seller on time)').mean():.0%}** are the carrier's "
                   "doing. Item problems come from keyword rules on the review text, so they are a lower bound.")
else:
    st.caption("No 1 or 2 star reviews by this date.")

st.markdown("**Evidence: this seller's orders** (newest first). Review text is the customer's own words, in Portuguese.")
ev = so.sort_values("ts_purchase", ascending=False).head(300).copy()
ev["dispatch"] = np.select([ev.ts_carrier <= T, ev.ship_limit <= T], ["On time", "Not dispatched, deadline passed"], "Pending")
ev.loc[(ev.ts_carrier <= T) & (ev.late_dispatch == 1), "dispatch"] = "Late"
ev["delivery"] = np.select([ev.ts_delivered <= T, (ev.ts_carrier <= T) & (ev.ts_estimated < T)],
                           [np.where(ev.late_delivery == 1, "Late", "On time"), "Not delivered, promise passed"], "Pending")
ev.loc[ev.failed, ["dispatch", "delivery"]] = "Cancelled"
ev.loc[ev.ts_review > T, ["review_score", "review_text"]] = None
ev["unfair"] = ev.unjust_review & (ev.ts_review <= T)
tag_names = list(data.TAG_COLS)
ev["complaint"] = ev[list(data.TAG_COLS.values())].apply(lambda r: ", ".join(n for n, v in zip(tag_names, r) if v), axis=1)
ev.loc[ev.ts_review > T, "complaint"] = ""
st.dataframe(ev[["order_id", "ts_purchase", "category", "price", "dispatch", "delivery", "review_score", "unfair", "complaint", "review_text"]]
             .assign(order_id=ev.order_id.str[:8]), hide_index=True, width="stretch", column_config={
    "order_id": "Order", "ts_purchase": st.column_config.DatetimeColumn("Bought", format="DD MMM YYYY"),
    "category": "Category", "price": st.column_config.NumberColumn("Price (R$)", format="%.2f"),
    "dispatch": "Dispatch", "delivery": "Delivery", "review_score": "Stars",
    "unfair": st.column_config.CheckboxColumn("Unfair review"), "complaint": "Complaint (from text)",
    "review_text": st.column_config.TextColumn("Review", width="large")})
