import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import data, ui

T, reveal = ui.sidebar_controls()
o = data.orders()
st.title("Order Journey Funnel")
st.markdown("Each stage counts only orders that passed every previous stage (strictly conditional). "
            "The two prediction windows sit on the two biggest drops: **dispatch** and **delivery**.")

c1, c2, c3 = st.columns([2, 1, 1])
rng = c1.date_input("Purchase date range", value=(pd.Timestamp("2017-01-01").date(), pd.Timestamp("2018-06-30").date()),
                    min_value=pd.Timestamp("2017-01-01").date(), max_value=pd.Timestamp("2018-08-31").date())
cats = sorted(o.category.dropna().unique())
states = sorted(o.customer_state.dropna().unique())
cat = c2.selectbox("Category", ["All"] + cats)
state = c3.selectbox("Customer state", ["All"] + states)
start, end = (pd.Timestamp(rng[0]), pd.Timestamp(rng[1]) + pd.Timedelta(days=1)) if len(rng) == 2 else (pd.Timestamp("2017-01-01"), pd.Timestamp("2018-07-01"))
f = o[(o.ts_purchase >= start) & (o.ts_purchase < end)]
if cat != "All":
    f = f[f.category == cat]
if state != "All":
    f = f[f.customer_state == state]


def stages(d):
    return [("Placed", len(d)), ("Payment approved", int(d.st_approved.sum())),
            ("Dispatched on time", int(d.st_dispatched_on_time.sum())),
            ("Delivered on time", int(d.st_delivered_on_time.sum())), ("Rated 4 or 5 stars", int(d.st_good_review.sum()))]


st.plotly_chart(ui.funnel(stages(f)), width="stretch")
st.caption("Orders bought in the last weeks of the data may not be delivered yet; they show as drop offs. The default range ends in June 2018 for that reason.")

st.subheader("Who caused the late deliveries?")
late = f[(f.late_delivery == 1) & f.late_dispatch.notna()]
c = st.columns(3)
ui.stat(c[0], "Late deliveries", f"{len(late):,}")
ui.stat(c[1], "Seller dispatched late", ui.pct((late.late_dispatch == 1).mean()))
ui.stat(c[2], "Seller was on time, carrier was late", ui.pct((late.late_dispatch == 0).mean()))
st.markdown(f"**1 or 2 star rate:** {ui.pct(f[f.late_delivery == 0].low_review.mean())} on on time deliveries versus "
            f"**{ui.pct(f[f.late_delivery == 1].low_review.mean())}** on late ones. "
            f"**{int(f.unjust_review.sum()):,}** bad reviews landed on sellers who had dispatched on time.")

st.subheader("Stress test: Black Friday 2017")
st.markdown("The same funnel for the Black Friday fortnight (20 Nov to 3 Dec 2017) against the six normal weeks before it. "
            "Order volume tripled; dispatch and delivery broke first.")
bf = o[(o.ts_purchase >= "2017-11-20") & (o.ts_purchase < "2017-12-04")]
normal = o[(o.ts_purchase >= "2017-10-09") & (o.ts_purchase < "2017-11-20")]
fig = go.Figure()
for name, d, color in [("Normal weeks (9 Oct to 19 Nov)", normal, ui.BLUE), ("Black Friday fortnight", bf, ui.ORANGE)]:
    st_ = stages(d)
    fig.add_bar(x=[s[0] for s in st_][1:], y=[s[1] / st_[0][1] for s in st_][1:], name=name, marker=dict(color=color, cornerradius=4),
                hovertemplate="%{x}: %{y:.1%} of placed orders<extra>" + name + "</extra>")
fig.update_yaxes(tickformat=".0%", range=[0.5, 1])
st.plotly_chart(ui.style(fig, 320), width="stretch")

st.subheader("Payment friction")
p = f[f.ts_approved.notna()].assign(hours=lambda d: (d.ts_approved - d.ts_purchase).dt.total_seconds() / 3600)
pt = p.groupby("pay_type").agg(orders=("order_id", "size"), median_hours_to_approval=("hours", "median")).sort_values("orders", ascending=False)
st.dataframe(pt.reset_index().rename(columns={"pay_type": "Payment type", "orders": "Orders", "median_hours_to_approval": "Median hours to approval"}),
             hide_index=True, column_config={"Median hours to approval": st.column_config.NumberColumn(format="%.1f")})
st.caption("Boleto (a Brazilian bank slip) takes about a day to clear; cards clear in minutes.")
