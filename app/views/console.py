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
if T >= data.PARTIAL_FROM:
    st.warning("Partial data: reviews stop on 31 Aug 2018 and the public sample only keeps reviewed orders, so orders from late "
               "August are mostly fast deliveries. Queues and outcomes for these days look better than reality.")

tab_d, tab_c, tab_s, tab_o = st.tabs(["Dispatch window", "Carrier window", "Seller silence", "Onboarding"])
sm = data.seller_months()
hist = sm[sm.month == data.panel_month_for(T)].set_index("seller_id")
TIER_ORDER = {"Overdue": 0, "Promise missed": 0, "High": 1, "Elevated": 2, "Normal": 3, "Not scored": 4, "Survey sent": 5}


def tier_help(window: str) -> str:
    r = data.tier_hit_rates(window)
    noun = "dispatched late" if window == "dispatch" else "arrived late or never"
    return (f"Tiers rank today's open orders. High: the riskiest 10%. Elevated: the next 20%. Over Sep 2017 to Aug 2018, "
            f"{r['High']:.0%} of High orders were {noun}, against {r['Elevated']:.0%} of Elevated and {r['Normal']:.0%} of Normal.")


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
                                         value=("price", "sum"), nearest_deadline=("days_to_deadline", "min"))
    s["open_orders"] = q.groupby("seller_id").size().reindex(s.index)
    s["past_late_rate"] = hist.late_dispatch_180.reindex(s.index)
    s["action"] = np.where(s.overdue > 0, "Call today: " + s.overdue.astype(int).astype(str) + " order(s) already past deadline",
                           "Call before " + (T + pd.to_timedelta(s.nearest_deadline.clip(lower=0), unit="D")).dt.strftime("%d %b")
                           + ": " + s.open_flagged.astype(str) + " order(s) likely to miss dispatch")
    s = s.sort_values(["overdue", "value"], ascending=False).reset_index()

    c = st.columns(4)
    ui.stat(c[0], "Orders awaiting dispatch", f"{len(q):,}")
    ui.stat(c[1], "Flagged (overdue or High)", f"{len(flagged):,}", tier_help("dispatch"))
    ui.stat(c[2], "Sellers to call", f"{len(s):,}")
    ui.stat(c[3], "Order value flagged", f"R$ {flagged.price.sum():,.0f}")
    st.markdown("**Call list**, ranked by overdue orders, then by order value at risk. The prediction is driven by the seller's "
                "dispatch habit (*past late rate*) and by what the seller is doing right now: open backlog, recent late handovers "
                "and a sudden surge in orders.")
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
        st.info(f"**Check against reality:** {ui.pct(done[done.flag].late_dispatch.mean())} of flagged orders were dispatched late "
                f"or never, versus {ui.pct(done[~done.flag].late_dispatch.mean())} of the orders not flagged.")
    with st.expander("Order level detail"):
        det = q.assign(_o=q.tier.map(TIER_ORDER)).sort_values(["_o", "p"], ascending=[True, False])
        det = det[["order_id", "seller_id", "ts_purchase", "ship_limit", "days_to_deadline", "tier", "category", "price"]
                  + (["late_dispatch"] if reveal else [])]
        st.dataframe(det.assign(order_id=det.order_id.str[:8], seller_id=det.seller_id.str[:8]), hide_index=True, width="stretch",
                     column_config={"tier": st.column_config.TextColumn("Risk tier", help=tier_help("dispatch")),
                                    "days_to_deadline": st.column_config.NumberColumn("Days to deadline", format="%.1f")})
    if len(s):
        open_seller(s.seller_id, "d_open")

    st.divider()
    st.markdown("#### Multi item orders: confirm before they ship")
    rates = data.shape_rates(on_time_only=False)
    multi = q[q.order_shape.isin(data.SHAPES[1:])].copy()
    st.markdown(f"Even when they arrive on time, orders with several items draw far more 1 or 2 star reviews: "
                f"**{rates.loc['2+ sellers', 'bad_rate']:.0%}** for items from two sellers and "
                f"**{rates.loc['2+ items, 1 seller', 'bad_rate']:.0%}** for several items from one seller, against "
                f"**{rates.loc['1 item', 'bad_rate']:.0%}** for a single item. The review text says why: one parcel arrives and the "
                "order shows as delivered. This is known at purchase, so no model is needed.")
    multi["shape_bad_rate"] = multi["order_shape"].map(rates.bad_rate)
    multi["action"] = np.where(multi["order_shape"] == "2+ sellers", "Tell the customer the items come in separate parcels",
                               "Ask the seller to ship all items together")
    multi = multi.sort_values(["shape_bad_rate", "days_to_deadline"], ascending=[False, True])
    mcols = ["order_id", "seller_id", "order_shape", "n_items", "n_sellers", "ship_limit", "shape_bad_rate", "action"]
    mview = multi[mcols].assign(order_id=multi.order_id.str[:8], seller_id=multi.seller_id.str[:8])
    if reveal:
        mview["stars"] = multi.review_score
    st.dataframe(mview, hide_index=True, width="stretch", column_config={
        "order_id": "Order", "seller_id": "Seller", "order_shape": "Order shape", "n_items": "Items", "n_sellers": "Sellers",
        "ship_limit": st.column_config.DatetimeColumn("Dispatch deadline", format="DD MMM"),
        "shape_bad_rate": st.column_config.ProgressColumn("Bad review rate for this shape", format="percent", min_value=0, max_value=1),
        "action": st.column_config.TextColumn("Recommended action", width="large"), "stars": "Stars"})
    if reveal:
        r = q[q.low_review.notna()]
        st.info(f"**Check against reality:** {ui.pct(r[r.order_shape != '1 item'].low_review.mean())} of these multi item orders got 1 or 2 "
                f"stars, versus {ui.pct(r[r.order_shape == '1 item'].low_review.mean())} of single item orders in the same queue.")

# Carrier window
with tab_c:
    q = data.carrier_queue(T)
    window = q[q.before_promise | q.promise_broken]
    f = window[window.flag].copy()
    lost = q[q.possibly_lost].copy()
    c = st.columns(4)
    ui.stat(c[0], "Parcels before their review survey", f"{len(window):,}",
            "With the carrier and the survey not yet sent. The survey goes out the day after delivery, or 2 days after the promised "
            "date if the parcel has not arrived (measured: true for 92% of delivered orders).")
    ui.stat(c[1], "High risk, promise still ahead", f"{(f.tier == 'High').sum():,}", tier_help("carrier"))
    ui.stat(c[2], "Promise already missed", f"{(f.tier == 'Promise missed').sum():,}",
            "Not delivered by the promised date, so late for certain. The KAM can still reach the customer before the survey.")
    ui.stat(c[3], f"Possibly lost ({data.LOST_FLAG_D}+ days, past promise)", f"{len(lost):,}",
            f"With the carrier {data.LOST_FLAG_D} or more days after handover and already past the promised date.")
    st.markdown("**Act before the survey date.** On late orders most customers review before the parcel arrives, so the review is "
                "decided when the promise breaks. Shock months (Black Friday, Mar 2018, the May 2018 strike) can raise the late rate "
                "for every parcel at once; the tiers still rank which parcels to call first.")
    f = f.assign(fault=np.where(f.seller_was_late, "Seller dispatched late", "Carrier (seller was on time)"),
                 lane=f.seller_state + " to " + f.customer_state,
                 action=np.select([f.tier == "Promise missed", f.seller_was_late],
                                  ["Promise missed: contact the customer before the survey; explain and give a new date",
                                   "Warn the customer; log the delay against the seller"],
                                  "Warn the customer; escalate the lane to logistics; protect the seller's rating"))
    f = f.assign(_o=f.tier.map(TIER_ORDER)).sort_values(["days_to_survey", "_o", "p"], ascending=[True, True, False])
    cols = ["order_id", "seller_id", "lane", "ts_estimated", "survey_date", "days_to_survey", "tier", "fault", "action"]
    view = f[cols].assign(order_id=f.order_id.str[:8], seller_id=f.seller_id.str[:8])
    if reveal:
        view["arrived_late"] = np.select([f.lost, f.late_delivery == 1, f.late_delivery == 0], ["Never arrived", "Yes", "No"], "Unknown")
        view["review"] = f.review_score
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "order_id": "Order", "seller_id": "Seller", "lane": "Lane (from to)",
        "ts_estimated": st.column_config.DateColumn("Promised date"), "survey_date": st.column_config.DateColumn("Survey goes out by"),
        "days_to_survey": st.column_config.NumberColumn("Days to survey", format="%d"),
        "tier": st.column_config.TextColumn("Status", help=tier_help("carrier")),
        "fault": "Cause so far", "action": st.column_config.TextColumn("Recommended action", width="large"),
        "arrived_late": "Arrived late", "review": "Stars"})
    if reveal:
        done = window[window.late_delivery.notna() & window.before_promise]
        st.info(f"**Check against reality:** {ui.pct(done[done.tier == 'High'].late_delivery.mean())} of High risk parcels arrived late "
                f"or never, versus {ui.pct(done[done.tier != 'High'].late_delivery.mean())} of the other parcels still before their promise.")
    lanes = f.groupby("lane").size().sort_values(ascending=False).head(10)
    if len(lanes):
        st.markdown("**Lanes with the most flagged parcels right now**")
        st.dataframe(lanes.rename("Parcels flagged").reset_index().rename(columns={"lane": "Lane"}), hide_index=True)

    st.divider()
    st.markdown(f"#### Possibly lost: {data.LOST_FLAG_D} or more days with the carrier, promise passed")
    curve = data.lost_curve().set_index("day").never_arrive
    st.markdown(f"Of parcels still with the carrier {data.LOST_FLAG_D} days after handover and already past their promised date, "
                f"**{curve.loc[20]:.0%}** never arrive; at 30 days **{curve.loc[30]:.0%}**, at 45 days **{curve.loc[45]:.0%}** "
                "(measured on 2017 to 2018 parcels). A rule on elapsed days, no model: the data has no tracking scans.")
    if len(lost):
        lost["chance_lost"] = data.lost_chance(lost.days_in_transit)
        lost["lane"] = lost.seller_state + " to " + lost.customer_state
        lost["action"] = np.where(lost.days_in_transit >= 30, "Open a carrier claim; offer a reship or refund now",
                                  "Ask the carrier for a trace; prepare a reship")
        lost = lost.sort_values("days_in_transit", ascending=False)
        lcols = ["order_id", "seller_id", "lane", "ts_carrier", "days_in_transit", "chance_lost", "action"]
        lview = lost[lcols].assign(order_id=lost.order_id.str[:8], seller_id=lost.seller_id.str[:8])
        if reveal:
            lview["outcome"] = np.where(lost.lost, "Never arrived", np.where(lost.ts_delivered.notna(),
                                        "Arrived " + lost.ts_delivered.dt.strftime("%d %b"), "Unknown"))
        st.dataframe(lview, hide_index=True, width="stretch", column_config={
            "order_id": "Order", "seller_id": "Seller", "lane": "Lane (from to)",
            "ts_carrier": st.column_config.DateColumn("Handed over"),
            "days_in_transit": st.column_config.NumberColumn("Days with carrier", format="%.0f"),
            "chance_lost": st.column_config.ProgressColumn("Chance it never arrives", format="percent", min_value=0, max_value=1),
            "action": st.column_config.TextColumn("Recommended action", width="large"), "outcome": "What happened"})
    else:
        st.caption("No parcel has been with the carrier that long on this date.")

# Seller silence
with tab_s:
    s = data.silence_list(T)
    w = s[s.window != "Stable"].sort_values("gmv_at_stake", ascending=False).copy()
    c = st.columns(4)
    ui.stat(c[0], "Sellers active in last 90 days", f"{len(s):,}")
    ui.stat(c[1], "Act now", f"{(s.window == 'Act now').sum():,}", "Predicted to stop selling starting next month.")
    ui.stat(c[2], "Window open", f"{(s.window == 'Window open').sum():,}", "Predicted to go silent within 3 months, not yet next month.")
    ui.stat(c[3], "Sales at stake (90 day GMV × risk)", f"R$ {w.gmv_at_stake.sum():,.0f}")
    st.markdown(f"Forecast as of month end **{data.panel_month_for(T)}**. Sellers with 10 or more orders in 90 days are scored by days "
                "since their last order, because for them that one signal ranks better than the model; everyone else gets the model. "
                "The warning signals explain the likely reason; they are not proven causes.")
    w["signals"] = w.apply(lambda r: "; ".join(data.signals(r)) or "No specific warning signal", axis=1)
    w["action"] = np.where(w.window == "Act now", "Check-in call this week", "Schedule a check-in this month")
    cols = ["seller_id", "window", "p_silent_1", "p_silent_3", "o_90", "g_90", "basis", "signals", "action"]
    view = w[cols].assign(seller_id=w.seller_id.str[:8])
    if reveal:
        view["went_silent"] = w.silent_by_3.map({1.0: "Yes", 0.0: "No"}).fillna("Unknown (after data end)")
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "seller_id": "Seller", "window": "Window",
        "p_silent_1": st.column_config.ProgressColumn("Silent next month", format="percent", min_value=0, max_value=1),
        "p_silent_3": st.column_config.ProgressColumn("Silent within 3 months", format="percent", min_value=0, max_value=1),
        "o_90": "Orders (90 days)", "g_90": st.column_config.NumberColumn("GMV 90 days (R$)", format="%.0f"),
        "basis": "Scored by", "signals": st.column_config.TextColumn("Warning signals", width="large"), "action": "Recommended action",
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
    decay = data.lead_decay().set_index("day").p_win
    st.markdown("Urgency comes from measured decay curves. Before signing: a new lead signs **"
                f"{decay.loc[0]:.0%}** of the time, but one still unsigned after 14 days only **{decay.loc[14]:.0%}**. After signing: the "
                "chance of a first sale by day 90 falls steadily the longer a seller waits. Offline and manufacturer sellers "
                "activate least often, so they get high touch support first.")
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

    st.divider()
    st.markdown("#### New sellers slow to ramp up")
    rq, young = data.ramp_queue(T)
    rule = data.ramp_rule().set_index("orders_30")
    st.markdown(f"Orders in the first 30 days after a seller's first sale are the clearest early sign of who becomes an established "
                f"seller (10 or more orders in days 31 to 180); a full model adds almost nothing to this count. With 1 order, "
                f"**{rule.loc[1, 'established_rate']:.0%}** get there; with 2, **{rule.loc[2, 'established_rate']:.0%}**; with 6 or more, "
                f"**{rule.loc[6, 'established_rate']:.0%}**. Listed: sellers 30 to 60 days after their first sale with 2 orders or fewer "
                f"in their first 30 days. {young:,} more sellers are still inside their first 30 days.")
    if len(rq):
        rq["action"] = "Growth call: review listings, prices and freight; agree a target for the next 30 days"
        rcols = ["seller_id", "first_sale", "days_since_first_sale", "orders_30", "p_established", "action"]
        rview = rq[rcols].assign(seller_id=rq.seller_id.str[:8])
        if reveal:
            rview["orders_31_180"] = np.where(rq.observed_180, rq.orders_31_180.astype(str), rq.orders_31_180.astype(str) + " (partly observed)")
        st.dataframe(rview, hide_index=True, width="stretch", column_config={
            "seller_id": "Seller", "first_sale": st.column_config.DateColumn("First sale"),
            "days_since_first_sale": "Days since first sale", "orders_30": "Orders in first 30 days",
            "p_established": st.column_config.ProgressColumn("Chance of becoming established", format="percent", min_value=0, max_value=1),
            "action": st.column_config.TextColumn("Recommended action", width="large"), "orders_31_180": "Orders in days 31 to 180"})
    else:
        st.caption("No seller is in the 30 to 60 day window with 2 orders or fewer on this date.")
