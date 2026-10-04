import numpy as np
import pandas as pd
import streamlit as st
from lib import data, ui

T, reveal = ui.sidebar_controls()
st.title("KAM Console")
st.caption(f"What the system knew on **{T:%A %d %b %Y}**. Order risks are re-scored every day an order sits in its window; "
           "models were trained only on outcomes known before this month.")
if T < pd.Timestamp("2017-09-01"):
    st.warning("Before September 2017 the models had less than six months of history to learn from, so predictions are weaker. "
               "Silence forecasts start in August 2017 and onboarding deals in December 2017.")

tab_d, tab_c, tab_s, tab_o = st.tabs(["Dispatch window", "Carrier window", "Seller silence", "Onboarding"])
sm = data.seller_months()
hist = sm[sm.month == data.panel_month_for(T)].set_index("seller_id")


def open_seller(options: pd.Series, key: str):
    """Pick a seller from this table and jump to the drilldown page."""
    c1, c2 = st.columns([3, 1])
    sid = c1.selectbox("Open a seller in the drilldown", options.unique(), format_func=lambda s: s[:8], key=key, index=None,
                       placeholder="Choose a seller ID")
    if c2.button("Open", key=key + "_btn", disabled=sid is None):
        st.session_state.seller = sid
        st.switch_page("views/seller.py")


# Dispatch window
with tab_d:
    q = data.dispatch_queue(T)
    flagged = q[q.flag]
    s = flagged.groupby("seller_id").agg(open_flagged=("order_id", "size"), overdue=("overdue", "sum"),
                                         value=("price", "sum"), nearest_deadline=("days_to_deadline", "min"),
                                         max_risk=("risk", "max"))
    s["open_orders"] = q.groupby("seller_id").size().reindex(s.index)
    s["past_late_rate"] = hist.late_dispatch_180.reindex(s.index)
    s["action"] = np.where(s.overdue > 0, "Call today: " + s.overdue.astype(int).astype(str) + " order(s) already past deadline",
                           "Call before " + (T + pd.to_timedelta(s.nearest_deadline.clip(lower=0), unit="D")).dt.strftime("%d %b")
                           + ": " + s.open_flagged.astype(str) + " order(s) likely to miss dispatch")
    s = s.sort_values(["overdue", "value"], ascending=False).reset_index()

    c = st.columns(4)
    ui.stat(c[0], "Orders awaiting dispatch", f"{len(q):,}")
    ui.stat(c[1], "Flagged (overdue or high risk)", f"{len(flagged):,}", f"High risk means predicted probability of {data.HIGH_DISPATCH:.0%} or more.")
    ui.stat(c[2], "Sellers to call", f"{len(s):,}")
    ui.stat(c[3], "Order value flagged", f"R$ {flagged.price.sum():,.0f}")
    st.markdown("**Call list**, ranked by overdue orders, then by order value at risk. "
                "The main driver of this prediction is the seller's own dispatch habit, shown as *past late rate*.")
    cols = ["seller_id", "action", "open_orders", "open_flagged", "overdue", "value", "past_late_rate"]
    view = s[cols].assign(seller_id=lambda x: x.seller_id.str[:8])
    if reveal:
        actual = flagged.groupby("seller_id").late_dispatch.agg(["sum", "count"])
        view["actually_late"] = (actual["sum"].astype(int).astype(str) + " of " + actual["count"].astype(str)).reindex(s.seller_id).values
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "seller_id": "Seller", "action": st.column_config.TextColumn("Recommended action", width="large"),
        "open_orders": "Open orders", "open_flagged": "Flagged", "overdue": "Overdue",
        "value": st.column_config.NumberColumn("Value flagged (R$)", format="%.0f"),
        "past_late_rate": st.column_config.ProgressColumn("Past late rate", format="percent", min_value=0, max_value=1),
        "actually_late": "Actually dispatched late"})
    if reveal:
        done = q[q.late_dispatch.notna()]
        st.info(f"**Check against reality:** {ui.pct(done[done.flag].late_dispatch.mean())} of flagged orders were dispatched late, "
                f"versus {ui.pct(done[~done.flag].late_dispatch.mean())} of the orders not flagged.")
    with st.expander("Order level detail"):
        det = q.sort_values("risk", ascending=False)[["order_id", "seller_id", "ts_purchase", "ship_limit", "days_to_deadline",
                                                      "risk", "category", "price"] + (["late_dispatch"] if reveal else [])]
        st.dataframe(det.assign(order_id=det.order_id.str[:8], seller_id=det.seller_id.str[:8]), hide_index=True, width="stretch",
                     column_config={"risk": st.column_config.ProgressColumn("Risk", format="percent", min_value=0, max_value=1),
                                    "days_to_deadline": st.column_config.NumberColumn("Days to deadline", format="%.1f")})
    if len(s):
        open_seller(s.seller_id, "d_open")

# Carrier window
with tab_c:
    q = data.carrier_queue(T)
    f = q[q.flag].sort_values("risk", ascending=False)
    c = st.columns(4)
    ui.stat(c[0], "Parcels with the carrier", f"{len(q):,}")
    ui.stat(c[1], "Likely to arrive late", f"{len(f):,}", f"Predicted probability of {data.HIGH_CARRIER:.0%} or more.")
    ui.stat(c[2], "Of those, seller dispatched on time", ui.pct((~f.seller_was_late).mean(), 0) if len(f) else "n/a",
            "If the seller was on time, this is a logistics problem: defend the seller, not blame them.")
    ui.stat(c[3], "Median days left before the promise", f"{f.days_to_promise.median():.0f}" if len(f) else "n/a")
    st.markdown("**Act before the promised date.** On late orders, most customers review before the parcel arrives, "
                "so the bad review is decided when the promise breaks.")
    f = f.assign(fault=np.where(f.seller_was_late, "Seller dispatched late", "Carrier (seller was on time)"),
                 lane=f.seller_state + " to " + f.customer_state,
                 action=np.where(f.seller_was_late, "Warn customer; log the delay against the seller",
                                 "Warn customer; escalate lane to logistics; protect the seller's rating"))
    cols = ["order_id", "seller_id", "lane", "ts_estimated", "days_to_promise", "risk", "fault", "action"]
    view = f[cols].assign(order_id=f.order_id.str[:8], seller_id=f.seller_id.str[:8])
    if reveal:
        view["arrived_late"] = f.late_delivery.map({1.0: "Yes", 0.0: "No"}).fillna("Not delivered")
        view["review"] = f.review_score
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "order_id": "Order", "seller_id": "Seller", "lane": "Lane (from to)",
        "ts_estimated": st.column_config.DateColumn("Promised date"),
        "days_to_promise": st.column_config.NumberColumn("Days left", format="%.0f"),
        "risk": st.column_config.ProgressColumn("Risk today", format="percent", min_value=0, max_value=1),
        "fault": "Cause so far", "action": st.column_config.TextColumn("Recommended action", width="large"),
        "arrived_late": "Arrived late", "review": "Stars"})
    if reveal:
        done = q[q.late_delivery.notna()]
        st.info(f"**Check against reality:** {ui.pct(done[done.flag].late_delivery.mean())} of flagged parcels arrived late, "
                f"versus {ui.pct(done[~done.flag].late_delivery.mean())} of the rest.")
    lanes = f.groupby("lane").size().sort_values(ascending=False).head(10)
    if len(lanes):
        st.markdown("**Lanes with the most at risk parcels right now**")
        st.dataframe(lanes.rename("Parcels at risk").reset_index().rename(columns={"lane": "Lane"}), hide_index=True)

# Seller silence
with tab_s:
    s = data.silence_list(T)
    w = s[s.window != "Stable"].sort_values("gmv_at_stake", ascending=False).copy()
    c = st.columns(4)
    ui.stat(c[0], "Sellers active in last 90 days", f"{len(s):,}")
    ui.stat(c[1], "Act now", f"{(s.window == 'Act now').sum():,}", "Predicted to stop selling starting next month.")
    ui.stat(c[2], "Window open", f"{(s.window == 'Window open').sum():,}", "Predicted to go silent within 3 months, not yet next month.")
    ui.stat(c[3], "Sales at stake (90 day GMV × risk)", f"R$ {w.gmv_at_stake.sum():,.0f}")
    st.markdown(f"Forecast as of month end **{data.panel_month_for(T)}**. Honest note: this forecast is driven mostly by "
                "how recently the seller sold. The warning signals below explain the likely reason; they are not proven causes.")
    w["signals"] = w.apply(lambda r: "; ".join(data.signals(r)) or "No specific warning signal", axis=1)
    w["action"] = np.where(w.window == "Act now", "Check-in call this week", "Schedule a check-in this month")
    cols = ["seller_id", "window", "p_silent_1", "p_silent_3", "o_90", "g_90", "signals", "action"]
    view = w[cols].assign(seller_id=w.seller_id.str[:8])
    if reveal:
        view["went_silent"] = w.silent_by_3.map({1.0: "Yes", 0.0: "No"}).fillna("Unknown (after data end)")
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "seller_id": "Seller", "window": "Window",
        "p_silent_1": st.column_config.ProgressColumn("Silent next month", format="percent", min_value=0, max_value=1),
        "p_silent_3": st.column_config.ProgressColumn("Silent within 3 months", format="percent", min_value=0, max_value=1),
        "o_90": "Orders (90 days)", "g_90": st.column_config.NumberColumn("GMV 90 days (R$)", format="%.0f"),
        "signals": st.column_config.TextColumn("Warning signals", width="large"), "action": "Recommended action",
        "went_silent": "Went silent within 3 months"})
    if reveal:
        k = s[s.silent_by_3.notna()]
        if len(k):
            st.info(f"**Check against reality:** {ui.pct(k[k.window != 'Stable'].silent_by_3.mean())} of flagged sellers went silent "
                    f"within 3 months, versus {ui.pct(k[k.window == 'Stable'].silent_by_3.mean())} of the rest.")
    if len(w):
        open_seller(w.seller_id, "s_open")

# Onboarding
with tab_o:
    st_ = data.stuck_sellers(T)
    c = st.columns(4)
    ui.stat(c[0], "Signed, not yet selling", f"{len(st_):,}")
    for i, stage in enumerate(["Setup call (day 15 to 44)", "Escalate (day 45 to 90)", "Likely lost (90+)"]):
        ui.stat(c[i + 1], stage, f"{(st_.stage == stage).sum():,}")
    st.markdown("Urgency comes from the measured decay curve: the chance of a first sale by day 90 falls steadily the longer a seller waits. "
                "Offline and manufacturer sellers activate least often, so they get high touch support first.")
    actions = {"Welcome (day 0 to 14)": "Welcome pack and listing checklist", "Setup call (day 15 to 44)": "Setup call: catalogue and first listing",
               "Escalate (day 45 to 90)": "Escalate: hands on onboarding session", "Likely lost (90+)": "Win back attempt or close the account"}
    st_["action"] = st_.stage.map(actions)
    st_ = st_.sort_values(["days_since_won"])
    cols = ["seller_id", "won_date", "days_since_won", "stage", "lead_type", "business_type", "business_segment", "p_sell_by_90", "action"]
    view = st_[cols].assign(seller_id=st_.seller_id.str[:8])
    if reveal:
        view["first_sale"] = st_.first_sale.dt.strftime("%d %b %Y").fillna("Never (by Aug 2018)")
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "seller_id": "Seller", "won_date": st.column_config.DateColumn("Deal won"), "days_since_won": "Days since won",
        "stage": "Stage", "lead_type": "Lead type", "business_type": "Business type", "business_segment": "Segment",
        "p_sell_by_90": st.column_config.ProgressColumn("Chance of selling by day 90", format="percent", min_value=0, max_value=1),
        "action": "Recommended action", "first_sale": "Actual first sale"})
