import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import data, ui

T, reveal = ui.sidebar_controls()
o = data.orders()
st.title("Why Bad Reviews")
st.markdown("The carrier window is built around late parcels, but lateness explains only part of the 1 and 2 star reviews. This page "
            "splits every bad review by what happened to the order, and reads the review text for the rest.")

r = o[o.low_review.notna() & o.clean & o.outcome.notna()]
low = r[r.low_review == 1]
split = low.outcome.value_counts().reindex(data.OUTCOMES).fillna(0)
rates = data.shape_rates(on_time_only=True)
on_time_low = low[low.outcome == "On time"]
with_text = on_time_low[on_time_low.review_text.fillna("").str.strip().str.len() > 3]
seller_side = with_text[["t_missing", "t_wrong", "t_defect"]].any(axis=1).mean()

c = st.columns(4)
ui.stat(c[0], "1 or 2 star reviews", f"{len(low):,}", "Orders promised by 29 Aug 2018 with a review.")
ui.stat(c[1], "On orders that arrived on time", ui.pct(split["On time"] / len(low), 0))
ui.stat(c[2], "Bad review rate, items from 2+ sellers", ui.pct(rates.loc["2+ sellers", "bad_rate"], 0),
        f"On time orders only. A single item order: {rates.loc['1 item', 'bad_rate']:.1%}.")
ui.stat(c[3], "On time bad reviews naming a wrong, defective or missing item", ui.pct(seller_side, 0),
        "Share of on time bad reviews with text. Keyword rules, so a lower bound.")

st.subheader("What happened to the orders behind bad reviews")
fig = go.Figure(go.Bar(x=split.values / len(low), y=split.index, orientation="h", marker=dict(color=ui.BLUE, cornerradius=4),
                       text=[f"{v / len(low):.1%}  ({int(v):,})" for v in split.values], textposition="outside", cliponaxis=False,
                       hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(tickformat=".0%", range=[0, split.max() / len(low) * 1.35])
st.plotly_chart(ui.style(fig, 260, legend=False), width="stretch")
rate_by = r.groupby("outcome").low_review.mean().reindex(data.OUTCOMES)
st.caption("Bad review rate by outcome: " + "; ".join(f"{k.lower()} {v:.0%}" for k, v in rate_by.items()) + ".")

st.subheader("Order shape: the problem a KAM can see at purchase")
left, right = st.columns([3, 2])
with left:
    fig = go.Figure(go.Bar(x=rates.index, y=rates.bad_rate, marker=dict(color=[ui.BLUE, ui.ORANGE, ui.CRITICAL], cornerradius=4),
                           text=[f"{v:.1%}<br>(n={n:,})" for v, n in zip(rates.bad_rate, rates.orders)], textposition="outside",
                           cliponaxis=False, hovertemplate="%{x}: %{y:.1%} get 1 or 2 stars<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", range=[0, rates.bad_rate.max() * 1.3], title="1 or 2 star rate, on time orders")
    st.plotly_chart(ui.style(fig, 320, legend=False), width="stretch")
with right:
    multi_share = rates.loc[data.SHAPES[1:], "orders"].sum() / rates.orders.sum()
    multi_bad = rates.loc[data.SHAPES[1:], "share_of_bad"].sum()
    st.markdown(f"Multi item orders are **{multi_share:.0%}** of on time orders but carry **{multi_bad:.0%}** of their bad reviews. "
                "Typical texts: *\"eu pedi dois kit e só chegou apenas um\"* (I ordered two kits and only one came) and *\"no site "
                "parece como foi entregue os dois\"* (the site shows both as delivered). The order appears "
                "to be marked delivered when the first parcel arrives (read from the reviews; Olist does not document it).")
    st.markdown("**KAM action:** flagged in the dispatch tab at purchase. Two sellers: tell the customer the items come separately. "
                "One seller: ask them to ship everything together.")

st.subheader("What customers say")
st.markdown("Share of bad reviews with text that mention each problem, by what happened to the order. A review can mention several.")
tw = low[low.review_text.fillna("").str.strip().str.len() > 3]
tags = pd.DataFrame({name: tw.groupby("outcome")[col].mean() for name, col in data.TAG_COLS.items()}).T
tags = tags.reindex(columns=[x for x in data.OUTCOMES if x in tags.columns])
counts = tw.outcome.value_counts()
tags.columns = [f"{x} (n={counts.get(x, 0):,})" for x in tags.columns]
tags.insert(0, "Rule precision", [data.TAG_PRECISION[k] for k in tags.index])
st.dataframe((tags * 100).round(0).astype(int).astype(str) + "%", width="stretch")
st.caption(f"{len(tw) / len(low):.0%} of bad reviews have text. Keyword rules in Portuguese, no model. Rule precision: share of 25 "
           "randomly drawn matches where the tag was right, read by hand. The rules are conservative: of 50 texts that matched no rule, "
           "38 described one of these problems in other words, so every share here is a lower bound.")

with st.expander("Read examples"):
    tag = st.selectbox("Complaint", list(data.TAG_COLS))
    ex = tw[tw[data.TAG_COLS[tag]]].sample(min(8, int(tw[data.TAG_COLS[tag]].sum())), random_state=1)
    for t, outcome in zip(ex.review_text, ex.outcome):
        st.markdown(f"> {t.strip()}  \n*{outcome}*")
