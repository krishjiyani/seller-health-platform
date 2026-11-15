# Iteration 2: data sanity, predictability map and plan

Drafted 10 Nov 2026. **Status 12 Nov 2026: approved; every P0 and P1 item is built in the pipeline and the app** (risk shown as tiers; the lost parcel flag also requires the promise to have passed, which doubles its precision). Items with no gain (F1 ETA and promise, F2 lane signals) were left out as planned. P2 is open. Every number comes from a script in `analysis/iter2/` (outputs in `analysis/iter2/out/`) run on the 11 raw CSVs. Rerun any script with `python -m analysis.iter2.<name>` from the repo root.

## 1. Where the project stands

Iteration 1 is complete and deployed as a Streamlit console: four intervention windows (dispatch, carrier, seller silence, onboarding), walk forward validation, an as of date replay with a "reveal what happened" check, and an Evidence page listing what failed. This session re-audited the data from scratch, checked the shipped scores, and tested seven new prediction targets.

The short version:

1. **Four data issues change iteration 1 numbers.** None breaks the story; two make it stronger.
2. **One model gets clearly better** (dispatch, with live seller signals). The carrier model's true AUC is 0.767, not 0.790, and live lane signals do not help it.
3. **Of seven new targets, two earn a place as rules, one as a new insight page, and the rest are documented negatives.**

## 2. Data sanity audit v2

| # | Finding | Evidence | Effect on iteration 1 | Fix |
|---|---|---|---|---|
| S1 | The promised delivery date has no time of day. Iteration 1 compares timestamps, so 1,291 parcels delivered on the promised day count as late | Late rate 8.13% (timestamp) vs 6.79% (calendar date). Those 1,291 have a 12.4% bad review rate, like on time orders (9.2%), not late ones (62.4%) | Late delivery label, carrier model, unfair review count, Overview tiles | Compare calendar dates |
| S2 | The review survey follows a fixed rule: the day after delivery, or 2 days after the promised date if the parcel has not arrived | Exact match for 92.7% of delivered orders (97.6% within a day), same in 2017 and 2018. Never delivered orders: 65.6% reviewed exactly at promise + 2 | Iteration 1 said the survey "appears" to fire near the promise. Now it is a measured rule, and the carrier window ends at promise + 2 | State the rule; count down to the survey date in the carrier tab |
| S3 | The public sample only holds orders with an answered review (99.2% have one), and reviews stop on 31 Aug 2018 | Orders bought 15 to 31 Aug 2018: 99.7% were delivered before 31 Aug. The week of 20 Aug has 1,071 orders vs about 1,900 normally; the week of 27 Aug has 130 | Late August is a fast delivery only sample. Its late rates and volumes are biased low | Train and evaluate only on orders promised by 29 Aug 2018 ("clean cohort"); mark late August as partial in the app |
| S4 | Lost parcels and orders never dispatched are left out of the labels | 1,104 orders handed to the carrier never arrived (69% bad reviews). 599 were never handed over (87% bad reviews). Every one passed its deadline before the data ends | The carrier model never sees its worst outcome. Positives are undercounted by 16.9% (carrier) and 6.9% (dispatch) | Count them as late |
| S5 | Promise policy changes over time | Median promise: 22 to 23 days in 2017, 27 in Jun 2018 (after the strike), 13.4 in Aug 2018. Monthly late rate ranges from 1.2% to 19% | Months are not interchangeable; probabilities drift | Walk forward already handles this; add recalibration |
| S6 | Deals only include wins from Dec 2017 | MQLs before Dec 2017: 2.8% win rate. MQLs from Dec 2017 on: 12.8% | Iteration 1's lead scoring AUC (0.68) used MQL timing, which was mostly this artifact | Use the Dec 2017 to May 2018 cohort only |
| S7 | Shipped scores are under calibrated | Carrier: mean predicted 10.9% vs actual 18.2%; orders scored 0 to 5% are really late 8.2% of the time. Dispatch: scored 2.6%, actual 8.0% at the low end | Risk percentages shown in the console read too low, worst in shock months | Recalibrate on recent resolved months, or show risk tiers instead of percentages |
| S8 | For big sellers the silence model loses to "days since last order" | Sellers with 10+ orders in 90 days: model 0.694 vs recency 0.713 (3 month horizon) | The model ranks the most valuable sellers worse than a one line rule | Use recency for big sellers; keep the model for small ones, where it wins slightly |
| S9 | Smaller quirks, low impact | 166 handovers before purchase (all Apr to Aug 2018); 1,359 handovers before payment approval; 19 deliveries before handover; 547 orders with several reviews; 789 review IDs reused on 1,412 orders; 610 products without category; 278 customers without geolocation | None material | List on the Evidence page |

Checks that passed: all 11 row counts; zero orphan keys; no won seller sold before the deal date; payments match item totals for 99.7% of orders; no approval before purchase; no delivery before purchase.

## 3. Headline numbers after correction

| Figure | Iteration 1 | Corrected | Why |
|---|---|---|---|
| Late delivery rate | 8.1% | **6.8%** | S1 |
| Bad review rate, late vs on time orders | 54.1% vs 9.2% | **62.4% vs 9.2%** | S1 (the link gets stronger) |
| Reviews on late orders written a full day or more before arrival | 62% | **74.5%** | S1 and S2 |
| Unfair bad reviews (seller on time, carrier late) | 2,983 | **2,867** | S1 |
| Late deliveries where the seller handed over on time | 72.8% | **72.3%** | S1 |
| Dispatch window AUC (monthly mean, daily rescored) | 0.763 | **0.768**, and **0.787** with live seller signals | S4, S3, F3 |
| Carrier window AUC (rule: days left) | 0.790 (rule 0.774) | **0.767 (rule 0.744)** | S1, S3, S4: the old figure was inflated by same day "late" parcels and the fast only August sample. The margin over the rule holds |
| Lead scoring AUC | 0.68 | **0.61** | S6 |

## 4. Predictability map

Every question the data can be asked about a seller or an order, with the verdict. "Window" is the time between when the prediction can be made and when the damage is done.

### Order lifecycle

| Stage | Question | Verdict | Evidence | Window |
|---|---|---|---|---|
| Purchase to dispatch deadline | Will the seller miss the dispatch deadline? | **Model earns its place** | 0.787 with live seller signals vs 0.723 for days waiting | About 6 days |
| Never dispatched | Will the order never be handed over? | Folded into the dispatch label (0.6% of orders) | S4 | Same |
| Handover to survey date | Will the parcel arrive after the promise? | **Model earns its place**, small margin over days left | 0.767 vs 0.744 for days left; shock months (Dec 2017, Mar 2018) are missed by 13 to 19 points in level | Promise + 2 days |
| Parcel lost in transit | Will it never arrive? | **Rule, no model** | In transit at day 20: 10.7% never arrive; day 30: 27.2%; day 45: 61.2% | Day 20 to 30 after handover |
| Delivery date | When will it arrive? | **No better than the lane median** | MAE 4.96 days vs 5.25 lane median; worse in shock months | Not worth building |
| Promise length | Could the promise be shorter at the same late rate? | **No** | A P95 promise gives 8.1% late at 22.0 days; Olist gives 7.8% at 22.8 | Not worth building |
| Bad review on an on time order | Will the customer complain anyway? | **Rule beats model**: order shape | 1 item: 7.5% bad; 2+ items from 2 sellers: 45 to 51%. Model at purchase 0.66 | Purchase to delivery |
| Bad review at purchase | Any bad review? | Not predictable | 0.63 | |
| Cancellation | Will the order be cancelled? | Not predictable | 0.61; cancel time not recorded | |
| Repeat purchase | Will the customer buy again? | Not predictable | 3.1% of people buy twice | |

### Seller lifecycle

| Stage | Question | Verdict | Evidence | Window |
|---|---|---|---|---|
| Lead to deal | Which leads will sign? | **Not deployable** | 0.61 (origin only) on a clean time split | |
| Lead to deal | When does a lead go cold? | **Descriptive window, strong** | Eventually win: 12.8% at day 0, 6.6% if not won by day 14, 4.4% by day 30, 2.7% by day 60. Median close 13 days | Day 0 to 14 |
| Deal to first sale | Which signed sellers will activate? | Not predictable from signup fields | 0.61; 92% of signup fields empty | |
| Deal to first sale | How fast does the chance fade? | Descriptive window (iteration 1) | 40% at day 15 down to 18% at day 60 | Day 0 to 44 |
| Sales rep effect | Do some reps' sellers activate more? | **No evidence** | 29% to 53% across 12 reps, chi square p = 0.29 | |
| First sale to established | Will a new seller ramp up? | **Rule beats model** | At day 30: model 0.860 vs orders in first 30 days 0.849. 1 order: 7% become established; 3: 41%; 6+: 86% | Day 0 to 30 after first sale |
| Active to silent | Will the seller stop selling? | Model for small sellers, recency for big ones | S8 | 1 to 3 months |
| Established decline | Will a big seller shrink? | Not predictable | 0.61, below recency 0.64 | |
| Exit reason | Why did the seller leave? | Not recorded | 62% of exits are abrupt | |
| Bad reviews to sales | Do bad reviews cost the seller orders? | **No significant effect** | Carrier caused bad review vs good review weeks: -5.5% orders, 95% CI [-12.1%, +1.6%] | |

## 5. What is left to predict: the seven feasibility tests

| Test | Question | Result | Verdict |
|---|---|---|---|
| F1 | ETA and promise | MAE 4.96 vs lane median 5.25; P95 promise matches Olist's late rate at almost the same length | Do not build. Show the lane median as "typical arrival" |
| F2 | Carrier model with corrected labels and live lane congestion | Corrected labels: 0.793 to **0.767** (rule 0.744). Live lane signals: 0.769, better in only 5 of 12 months, calibration worse | Keep iteration 1's lean features on corrected labels. Do not adopt lane signals. Show tiers, not percentages |
| F3 | Dispatch model with live seller signals | 0.768 to **0.787**, better in 12 of 12 months; calibration gap 0.044 to 0.036 | **Adopt** |
| F4 | Lead scoring and lead decay | Scoring 0.61; decay curve strong; no sales rep effect | Add the decay window; no scoring model |
| F5 | New seller ramp | 0.860 vs 0.849 for a count rule | Add as a rule table and a console queue |
| F6 | Bad reviews not about lateness; complaint taxonomy | 57.3% of bad reviews are on on time orders. Multi seller orders 45 to 51% bad. Keyword taxonomy tags 72.7% of bad review texts; strong tags about 88 to 90% precise on a spot check | **New insight page** and an order shape flag |
| F7 | Do bad reviews cost sales | -5.5%, CI [-12.1%, +1.6%] | Evidence page finding |

### Why F6 is the most important new result

Iteration 1 built the carrier window around late deliveries. But only 27.7% of all 1 to 2 star reviews are on late orders. **57.3% are on orders that arrived on time**, and the rest are cancellations (6.4%), lost parcels (5.0%) and orders never dispatched (3.6%). On on time orders the bad review rate depends on the order's shape:

| On time orders | Bad review rate | Orders |
|---|---|---|
| 1 item, 1 seller | 7.5% | 80,174 |
| 2 items, 1 seller | 19.7% | 5,982 |
| 3+ items, 1 seller | 25.3% | 1,780 |
| 2 items, 2 sellers | 45.4% | 934 |
| 3+ items, 2+ sellers | 51.3% | 312 |

Multi item orders are about 10% of on time orders but carry about 27% of on time bad reviews. The review text says why: "only one of the two arrived", "the site says both were delivered". The order appears to be marked delivered when the first parcel arrives (inferred from the text, not documented). This is a fulfilment problem a KAM can act on, it is known at purchase, and it needs no model.

## 6. Iteration 2 plan

Priorities: **P0** fixes correctness and must ship; **P1** adds the strongest new value; **P2** is optional.

### P0. Correctness patch (pipeline, about 1 session)

1. Late delivery by calendar date (S1).
2. Count lost parcels as late deliveries and never dispatched orders as late dispatches (S4).
3. Clean cohort for training and evaluation: promised by 29 Aug 2018. The app shows late August with a "partial data" note (S3).
4. Carrier window ends at the survey date (promise + 2); the console counts down to it (S2).
5. Recalibrate both order models on the last resolved months (isotonic). Acceptance: mean absolute calibration gap under 0.02 per month on the test months. If that fails, show tiers (High, Elevated, Normal) instead of percentages (S7).
6. Silence: rank sellers with 10+ orders in 90 days by recency; keep the model for the rest (S8).
7. Lead funnel numbers from the Dec 2017 cohort only (S6).
8. Rerun every headline number; update README, the Evidence page and `PREDICTABILITY_AUDIT.md` with a "corrections in iteration 2" section. The mistakes caught are viva material, like the two caught in iteration 1.

Acceptance: golden seller test still passes; new assertions on the late rate under the date rule (6.5% to 7.0%) and on the survey rule match rate (over 90%).

### P1. Model upgrade and new modules (about 2 to 3 sessions)

1. **Dispatch model with live seller signals** (F3): open backlog, late share over the last 14 days, handover speed over the last 30 days, surge ratio. Acceptance: at least 0.78 monthly mean AUC and better than iteration 1 in at least 10 of 12 months.
2. **Carrier model**: retrain iteration 1's lean features on the corrected labels; no live lane signals (F2 showed no gain). Carrier shocks are not foreseeable in this data, so the carrier tab shows tiers.
3. **"Why bad reviews" page** (F6): where bad reviews come from (on time, late, lost, cancelled, never dispatched); the order shape table; the complaint taxonomy by outcome; per seller split of seller owned and carrier owned complaints in the Seller Drilldown. Before shipping, tighten the "late" rule (negation: "no prazo", "antes do prazo") and drop the "service" tag. Acceptance: hand label 200 bad reviews and report precision per tag; ship only tags at 80% precision or better.
4. **Order shape flag** in the dispatch tab: multi item and multi seller orders flagged at purchase with "confirm all items ship together, or tell the customer". Acceptance: replay shows the flag's bad review rate against the rest.
5. **Lost parcel flag** in the carrier tab: still in transit 20 days after handover means "open a carrier claim, offer a reship".
6. **New seller ramp queue** (F5) in the onboarding tab: sellers 0 to 60 days after first sale with 1 or 2 orders in their first 30 days, with the rule table as the reason.
7. **Lead decay** (F4) at the top of the Onboarding page: leads go cold after 14 days.

### P2. Optional

1. **KAM capacity planner**: given N calls a day, pick the sellers to call by expected late orders prevented, not only by overdue count. Measured by replay: share of late orders that fall on called sellers, against the current ranking.
2. **Evidence additions**: calibration plots, the F7 natural experiment, the sales rep null result, the lead truncation story, the survey rule chart.
3. **Docs**: synopsis wording (scikit learn instead of LightGBM; the window framing), poster numbers.

## 7. What we will not build, and why

| Idea | Why not |
|---|---|
| ETA or a shorter promise | No gain over a lane median; Olist's promise already sits at the right quantile (F1) |
| Lead scoring model | 0.61 on a clean split (F4) |
| Sales rep leaderboard | Differences are within chance (F4) |
| Established seller decline predictor | 0.61, below recency (iteration 1) |
| Bad review prediction at purchase | 0.63 overall, 0.66 on on time orders; the order shape rule carries most of it (F6) |
| Demand or GMV forecasting | 20 months of data and one Black Friday; no yearly seasonality to learn |
| Price elasticity or conversion | No data on visits or products not bought |
| "Unfair reviews kill sellers" claim | Not supported twice (strike test; F7) |

## 8. Future scope beyond this data

What would unlock more, for the report's future scope section:

* **Intervention logs** (who called whom, when, and what happened next). This is the only route to showing that acting helps (uplift modelling or an A/B test).
* **Cancellation timestamps**, so cancelled orders can stay in live queues.
* **Listings, stock and catalogue changes**, which would show a seller pausing before the orders stop. Today 62% of exits look abrupt.
* **Carrier identity and tracking scans**, for a real lost parcel model.
* **Visits and add to cart events**, for conversion and the effect of the promise length.
* **Marketplace channel** (the reviews name channels only by alias, and only when customers mention them).

## 9. Streamlit changes (only after approval)

| Page | Change |
|---|---|
| Overview | Corrected headline tiles; add "57% of bad reviews are on on time orders" |
| KAM Console, Dispatch | Live signal model; order shape flag; calibrated risk or tiers |
| KAM Console, Carrier | Countdown to the survey date; lost parcel flag; typical arrival from the lane median |
| KAM Console, Silence | Recency ranking for big sellers |
| KAM Console, Onboarding | Ramp queue for new sellers; lead decay note |
| Seller Drilldown | Seller owned vs carrier owned complaints |
| New page: Why Bad Reviews | Outcome split, order shape table, complaint taxonomy |
| Onboarding funnel | Lead decay curve; clean cohort |
| Evidence and Limits | Corrections log, calibration plots, F7, sales rep test, survey rule |

## 10. Decisions needed

1. Approve the P0 correctness patch. It changes numbers already on the synopsis and poster.
2. Risk display: calibrated percentages, or tiers?
3. Which P1 items to build, and in what order. Recommended: P0, then F3 dispatch upgrade, then the Why Bad Reviews page, then the flags and queues.
4. Deadline for the next review or viva, which sets how much of P2 fits.
