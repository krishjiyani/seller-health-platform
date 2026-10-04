import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import data, ui

T, reveal = ui.sidebar_controls()
o = data.orders()
st.title("Seller Lifecycle Funnel")
st.markdown("Every month, each seller who has ever sold sits in one state. **New**: first sale this month. **Active**: sold this month. "
            "**Slowing**: no sale this month but sold in one of the two months before. **Silent**: no sale for three months or more.")

STATES = ["New", "Active", "Slowing", "Silent"]


@st.cache_data
def states() -> pd.DataFrame:
    x = o[o.seller_id.notna()]
    months = pd.period_range("2017-01", "2018-08", freq="M")
    c = x.groupby(["seller_id", x.ts_purchase.dt.to_period("M")]).size().unstack(fill_value=0).reindex(columns=months, fill_value=0)
    A = c.values > 0
    first = A.argmax(1)
    out = np.full(A.shape, "", dtype=object)
    for j in range(len(months)):
        prev2 = A[:, max(0, j - 2):j].any(1) if j else np.zeros(len(A), bool)
        out[:, j] = np.where(first > j, "", np.where(first == j, "New", np.where(A[:, j], "Active", np.where(prev2, "Slowing", "Silent"))))
    return pd.DataFrame(out, index=c.index, columns=[str(m) for m in months])


S = states()
counts = S.apply(lambda col: col.value_counts()).reindex(STATES).fillna(0).T
fig = go.Figure()
for state, color in zip(STATES, ui.SERIES):
    fig.add_bar(x=counts.index, y=counts[state], name=state, marker=dict(color=color),
                hovertemplate="%{x}: %{y:,.0f} sellers<extra>" + state + "</extra>")
fig.update_layout(barmode="stack")
st.plotly_chart(ui.style(fig, 360), width="stretch")
st.caption("The silent group only grows: once a seller stops, few come back.")

st.subheader("Where sellers moved, month to month")
cols = list(S.columns)
c1, c2 = st.columns(2)
a = c1.selectbox("From month", cols[:-1], index=cols.index("2018-02"))
b = c2.selectbox("To month", cols[cols.index(a) + 1:], index=0)
pair = S[(S[a] != "")][[a, b]]
mat = pd.crosstab(pair[a], pair[b], normalize="index").reindex(index=STATES, columns=["Active", "Slowing", "Silent"]).dropna(how="all")
st.dataframe((mat * 100).round(1).rename_axis(index=f"State in {a}", columns=f"State in {b} (% of row)"), width="stretch")

st.subheader("What the data says about seller exits")
st.markdown(
    "* Of sellers who stopped after at least six months of history, **62% stopped abruptly**: their last three months looked like "
    "the three before. Only **23%** declined gradually first.\n"
    "* So for most exits there is no warning in the data; the reason a seller leaves is not recorded.\n"
    "* The silence forecast in the KAM Console therefore catches sellers who are **already slowing**, which is honest and still useful "
    "for triage, but it cannot see an abrupt exit coming.")
