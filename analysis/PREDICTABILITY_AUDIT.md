# Predictability Audit: what the Olist data can and cannot predict

Run on 2026-10-02 against the 11 Kaggle CSVs in `data/raw/`. Every number below comes from the scripts in this folder (`audit_01` to `audit_08`) and is reproducible.

## 0. Data verification

All 11 tables match the expected row counts exactly. Zero orphan keys on items to orders, items to sellers, and deals to MQL. Headline facts reproduce: late handover 9.03%, late delivery 8.11%, 27.2% of late deliveries had a late handover. New fact: `won_date` only starts in Dec 2017, so funnel sellers have at most 9 months of order history.

## 1. Method

* **Seller targets:** a panel of 19,890 seller months (2,905 sellers, origins Mar 2017 to Jul 2018). Features use only information timestamped on or before the origin date T. Labels come from the months after T.
* **Walk forward validation:** each test month is scored by a model trained only on origins whose label window had already closed. That gives 7 to 9 monthly folds per target.
* **Order targets:** trained on purchases from Feb to Dec 2017 and tested on Jan to Jun 2018 (about 40,000 orders). Seller history features come from the previous month end.
* **Model:** gradient boosted trees (scikit learn), fixed seed.
* **Baseline:** every seller target is also scored by "days since last order" alone. A model that barely beats this baseline is not adding value.

## 2. Results

### 2.1 Seller level

| Target | Base rate | Model AUC | Recency only AUC | Verdict |
|---|---|---|---|---|
| Dormant next 90 days (all active sellers) | 23.5% | **0.83** | 0.79 | Predictable, but mostly recency |
| Lost next 90 days (dormant OR GMV drop >40%) | 46.5% | 0.67 | 0.65 | Weak; label too noisy for small sellers |
| GMV drop >40% among non dormant | 30.1% | 0.66 | 0.51 | Weak |
| Silence starts within 1 / 2 / 3 months (survival horizons) | 24 / 30 / 35% | 0.83 / 0.81 / 0.79 | 0.79 / 0.77 / 0.76 | Survival curve is feasible |
| Dormant, new sellers (tenure 60 days or less) | 27.7% | 0.74 | 0.72 | Moderate |
| Decline, established sellers (10+ orders in 90 days) | 35.8% | 0.61 | 0.64 | **Not predictable beyond recency** |
| Next 90 day GMV (rank correlation) | | 0.70 | 0.64 (last 90 day GMV) | Small gain |

Ablation for the lost target: activity features alone give 0.662 AUC; operational and quality pillars alone give 0.578.

### 2.2 Shape of seller loss (is there a window?)

Of 339 observed losses with at least 6 months of history:

* **61.7% were abrupt stops:** the last 3 months had as many orders as the 3 months before. No early signal exists for these.
* **23.3% were gradual declines:** the last 3 months had under 60% of the previous 3.
* For sellers with 20+ lifetime orders, the figures are 42.5% abrupt and 30.1% gradual.

**Conclusion:** a seller decline window exists for roughly one loss in four. For the majority the data records silence, not a slide.

### 2.3 Doom loop (correlational)

Among established sellers, grouped into quartiles within their volume band, the worst quartile on each signal retained less GMV over the next 90 days:

| Signal (worst vs best quartile) | Median GMV retained | Drop >40% rate |
|---|---|---|
| Late handover rate | 0.73 vs 0.97 | 42% vs 29% |
| Share of 1 to 2 star reviews | 0.71 vs 0.90 | 42% vs 31% |
| Cancel/unavailable rate | 0.66 vs 0.99 | 46% vs 27% |
| Orders past dispatch deadline right now | 0.49 vs 1.00 | 58% vs 28% |

The association is real, but it is not strong enough to rank individual established sellers well (section 2.1). These signals work as **explanations and playbook triggers**, not as a predictor.

### 2.4 Order level

| Target | Decided at | Window length | Base rate | AUC | Top 10% hit rate |
|---|---|---|---|---|---|
| Late seller handover | Purchase | 6 days median until dispatch deadline | 8.4% | **0.70** | 21.6% (2.6x) |
| Late customer delivery | Purchase | | 10.0% | 0.65 | 20.8% |
| Late customer delivery | Carrier handover | **20 days median** until promised date | 10.0% | **0.72** | 27.7% (2.8x) |
| 1 to 2 star review | Purchase | | 15.7% | 0.63 | 31.3% |
| 1 to 2 star review | After delivery (knows delays) | ~0 days (see below) | 15.7% | 0.80 | 71.8% |
| Cancel/unavailable | Purchase | | 0.4% | 0.61 | 1.0% |
| Customer buys again within 180 days | First order | | 1.7% | 0.57 | 2.3% |

**Critical timing fact:** judging from timestamps, the review survey appears to fire at delivery or near the promised date, whichever comes first. The median review is written 0.2 days after delivery. On late orders, **62% of reviews are dated a full day or more before the parcel arrives** (69% if same day reviews are counted; the review date has no time of day, so 62% is the figure to quote). So the "after delivery" model (0.80) has no actionable window. The real window is **carrier handover to the promised date (median 20 days)**: act before the promise breaks and the survey fires.

Supporting facts: 1 to 2 star rate is 9.2% on on time orders and 54.1% on late ones; 33.7% of all 1 to 2 star reviews are on late orders; **2,983** 1 to 2 star reviews sit on orders that were late even though the seller handed over on time (reputation damage the seller did not cause).

### 2.5 Leads and activation

| Target | n | Base rate | AUC | Verdict |
|---|---|---|---|---|
| MQL to won (origin, landing page, timing) | 8,000 | 10.5% | 0.68 ± 0.01 | Moderate |
| Won to first sale within 90 days (signup attributes) | 665 | 43.9% | 0.61 ± 0.04 | Not deployable (confirms earlier test) |

Activation decays with time, which defines the onboarding window without any model:

| Not sold by day | Sellers | Will still sell by day 90 |
|---|---|---|
| 15 | 618 | 39.6% |
| 30 | 554 | 32.7% |
| 44 | 505 | 26.1% |
| 60 | 453 | 17.7% |

Activation lag: median 46.5 days, p75 76, p90 105.

**Cohort note:** this test uses deals won on or before 1 Jun 2018 and counts a first sale within 90 days. The synopsis figures (51.9% activation, 44 day median lag, 48% never activate) used deals won at least 90 days before 31 Aug 2018 and counted any first sale. Both are correct for their definitions; the report must state which one it quotes.

### 2.6 What drives the order window models (`audit_07`)

Permutation importance: the AUC lost when one input is shuffled on 8,000 test orders.

| Window | Main drivers | Single input baselines |
|---|---|---|
| Dispatch (late handover, at purchase) | Seller's own past late handover rate (0.168); everything else under 0.01 | Past late rate alone: AUC **0.71**, equal to the full model (0.706) |
| Carrier (late delivery, at handover) | Destination state (0.094), promised days (0.069), handover delay (0.068), month (0.018), distance (0.008) | Promised days alone 0.55; handover delay alone 0.63; full model **0.72** |

**Reading:** dispatch lateness is a persistent seller habit, so the dispatch prediction is essentially a seller reliability score. That makes it fully explainable. The carrier prediction genuinely combines several factors.

### 2.7 Brazilian public holidays (`audit_07`)

13.7% of orders had a public holiday inside their dispatch window. Late handover rate: 9.2% with a holiday versus 9.0% without. Adding holiday features changed AUC from 0.706 to 0.696 for dispatch and from 0.721 to 0.720 for carrier. **No effect. Dropped.**

### 2.8 Natural shocks (`audit_07`, `audit_08`)

**Black Friday 2017 (demand shock):**

| Week | Orders | Late handover | Late delivery | 1 to 2 star | Median transit (days) |
|---|---|---|---|---|---|
| 30 Oct to 5 Nov | 921 | 7.8% | 6.3% | 13.0% | 7.1 |
| 20 to 26 Nov (Black Friday) | 2,979 | 15.3% | 19.4% | 21.3% | 9.9 |
| 27 Nov to 3 Dec | 2,078 | 14.1% | 16.7% | 19.9% | 9.5 |
| 11 to 17 Dec | 1,398 | 8.4% | 6.3% | 15.9% | 9.2 |

**Peak stress test:** 179 sellers with at least 10 orders before Black Friday (1 Sep to 19 Nov) and at least 5 during it (20 Nov to 3 Dec). 37 "broke": their late handover rate rose by 15 points or more. Next quarter (Jan to Mar 2018), compared with the 142 that held:

| | Held (142) | Broke (37) |
|---|---|---|
| Median GMV growth (per day, versus before Black Friday) | 1.11x | 0.93x |
| GMV drop over 40% | 22.5% | 35.1% |
| Fully dormant | 0.0% | 8.1% |

The direction is consistent, but the sample is small and the predictive power modest (AUC 0.59 for the drop). **Use as an insight with a sample size caveat, not as a predictor.**

**Truckers' strike, 21 May to 1 Jun 2018 (carrier shock):** weekly orders fell from about 1,900 to 951, and median transit rose from about 7 to 9.5 days. 215 sellers had at least 5 orders dispatched on time between 7 and 27 May. For the 38 where 20% or more of those orders still arrived late:

| | Not hit (177) | Hit (38) |
|---|---|---|
| Carrier late rate | 4.7% | 28.6% |
| 1 to 2 star rate | 8.8% | 14.2% |
| GMV growth Jun to Jul versus Feb to Apr (per day) | 0.88x | 1.03x |

**Reading:** carrier delays produce unfair bad reviews (reputation damage the seller did not cause), but no measurable short term sales loss. The original "unjust death" hypothesis is therefore **not supported**. The supported claim is **"unjust reviews"**.

### 2.9 Second dataset search (2 Oct 2026)

| Dataset | Verdict |
|---|---|
| Tmall Repeat Buyers (Alibaba Tianchi, 55M events, merchant IDs, click/cart/purchase/favourite) | Only real public dataset with a pre purchase funnel per merchant. China 2015, no delivery or reviews, cannot be joined to Olist, task is customer repurchase. Different project; not adopted |
| MerRec (Mercari, Hugging Face) | Schema checked: no seller identifier, no delivery timestamps, no reviews. Rejected |
| Synthetic US ecommerce (Kaggle, 1M rows) | Synthetic; findings would not be real. Rejected |
| DataCo Smart Supply Chain | Single company, no sellers, 55% late base rate, label reported as derivable from shipping day columns. Rejected |
| Cainiao LaDe | Real courier data, no sellers or reviews. Future scope reference only |
| REES46, OTTO, RetailRocket, Coveo | Sellers anonymised (earlier search) |

**Conclusion:** Olist plus the marketing funnel remains the only real public dataset linking sellers, order timestamps and reviews.

## 3. What this means for the product

**Predictable with an actionable window:**

1. **Dispatch window** (purchase to dispatch deadline, about 6 days): flag orders likely to miss handover. AUC 0.70.
2. **Carrier window** (handover to promised date, about 20 days): flag orders likely to arrive late before the review survey fires. AUC 0.72.
3. **Onboarding window** (won to day 30 to 44): the chance of a first sale by day 90 falls from 40% for a seller still unsold at day 15 to 18% for one still unsold at day 60. Descriptive, no model needed.
4. **Seller dormancy survival curve** (1 to 3 months): AUC 0.79 to 0.83, but largely a recency signal. Useful for triage and a survival view, not a deep insight.

**Not predictable (state as findings):** established seller decline beyond recency (0.61), day zero activation (0.61), cancellations (0.61), repeat purchase (0.57), review score at purchase (0.63), abrupt seller exits (62% of losses show no prior decline).

**Limitations:** the data contains no interventions, so no result here says acting helps. The survey timing claim is inferred from timestamps, not from Olist documentation.

## 4. How the predictions are made

1. **Examples from history.** Each past order (or seller month) becomes one row. The inputs are only what was known at that moment; the answer is computed from timestamps. Example: `late_handover = 1 if order_delivered_carrier_date > shipping_limit_date`.
2. **Train on the past, test on the future.** Order models train on Feb to Dec 2017 purchases and are tested on Jan to Jun 2018. Seller models use walk forward validation: each test month is scored by a model trained only on months whose outcomes were already known.
3. **Model.** Gradient boosted decision trees (scikit learn `HistGradientBoostingClassifier`, fixed seed). About 300 shallow trees are added in sequence, each correcting the previous ones. The summed score F becomes a probability through `p = 1 / (1 + e^(-F))`. Training minimises log loss.
4. **Measurement.** AUC: the probability that a randomly chosen positive case scores higher than a randomly chosen negative one (0.5 is chance). Also reported: the hit rate in the top 10 or 20% of scores versus the base rate (lift), and a simple single input baseline for every target.
5. **Model choice.** LightGBM (named in the synopsis) needs a system library (libomp) that is unavailable on the build machine. scikit learn implements the same algorithm family with no system dependency, so it is used everywhere.

## 5. Funnels adopted for iteration 1

Every stage is defined from a timestamp in the data. Stages are strictly conditional: each stage counts only units that passed the previous one.

1. **Order journey:** placed, approved, dispatched on time, delivered on time, rated 4 to 5 stars. The dispatch window and the carrier window sit on the two largest drops. The Black Friday week is shown as a stress comparison.
2. **Seller onboarding:** lead, won deal, first sale, still selling in month 1. The activation decay curve (section 2.5) sets call urgency.
3. **Seller lifecycle:** active, slowing, silent. The 1, 2 and 3 month silence forecast, labelled as largely a recency signal.

## 6. Positioning

**Title (unchanged):** Seller Health and Intervention Platform.
**Optional subtitle:** Predicting Intervention Windows Across the Seller and Order Lifecycle in Ecommerce Marketplaces.
**Pitch:** a decision support console that tells a key account manager which orders and sellers need action this week, built on measured intervention windows, with every prediction validated on future data and every untestable or failed hypothesis documented.
**Avoid:** claiming the system predicts which sellers will fail. Established seller decline is not predictable beyond recency (section 2.1), and 62% of exits show no prior signal (section 2.2).
