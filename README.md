# Seller Health and Intervention Platform

A decision support console for key account managers (KAMs) on an ecommerce marketplace. It predicts **when** to intervene, at order level and seller level, while there is still time to prevent the damage, and recommends the action. Built on the public Olist Brazilian marketplace data (Jan 2017 to Aug 2018).

Final year BTech project, Group BCAIAA61, MIT School of Computing, MIT ADT University, Pune. Guide: Prof. Sneha Singha.

## What it predicts

| Window | Question | Time to act | Model AUC | Best simple rule | High tier (riskiest 10% each day) |
|---|---|---|---|---|---|
| Dispatch | Will this order miss the seller's dispatch deadline, or never ship? | about 6 days | 0.79 | 0.72 (days waiting), beaten in 12 of 12 months | 66% go late, against 12% of Normal |
| Carrier | Will this parcel arrive after the promised date, or never? | handover to survey date | 0.77 | 0.74 (days left), beaten in 11 of 12 months | 49% go late, against 9% of Normal |
| Silence | Will this seller stop selling within 1 to 3 months? | 1 to 3 months | 0.82 to 0.79 | 0.79 to 0.76 (recency) | |
| Onboarding | Will a lead, a signed seller or a new seller go cold? | day 0 to 14 (leads), 0 to 44 (signed), 30 to 60 (new sellers) | measured decay curves and a day 30 rule | | |

AUC is averaged over monthly walk forward tests from Sep 2017 to Aug 2018: every month is scored by a model trained only on outcomes known before it. Order risks are re-scored every day an order stays inside its window, and the console shows tiers (High, Elevated, Normal) rather than percentages. The app also has a **Why Bad Reviews** page: 57% of 1 or 2 star reviews are on orders that arrived on time, and orders with items from two sellers draw them 46% of the time. Full results, including what the data cannot predict, are on the app's Evidence page and in `analysis/PREDICTABILITY_AUDIT.md`.

## What else the console does

* **Dispatch window:** the model reads the seller's dispatch habit and live signals (open backlog, late handovers in the last 14 days, a surge in orders). Orders with items from two sellers, or several items, are flagged at purchase.
* **Carrier window:** counts down to the review survey date; parcels past their promise are marked as certain to be late, and parcels 20+ days with the carrier and past the promise are flagged as possibly lost.
* **Silence:** sellers with 10+ orders in 90 days are scored by days since their last order, which ranks them better than the model.
* **Onboarding:** a lead decay curve (leads go cold after 14 days), the post signing decay curve, and a queue of new sellers with 2 orders or fewer in their first 30 days.
* **Why Bad Reviews:** bad reviews split by what happened to the order, the order shape effect, and complaint tags read from the review text.

## Project layout

```
app/                 Streamlit app (entry: app/streamlit_app.py, pages in app/views/)
pipeline/            raw CSVs to app tables: build.py (tables + assertions), models.py (walk forward models),
                     live.py (live seller signals), taxonomy.py (review complaint rules), run.py
tests/               golden seller check: recomputes sellers from raw CSVs by an independent route
analysis/            predictability audit scripts and write ups (data checks and feasibility tests in analysis/iter2/)
data/app/            precomputed tables the app reads (about 30 MB, committed)
data/raw/            the 11 Kaggle CSVs (not committed; download below)
docs/                synopsis and poster
```

## Run locally

```bash
uv venv --python 3.11 .venv          # or: python3.11 -m venv .venv
uv pip install -r requirements.txt   # or: .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app/streamlit_app.py
```

The app only needs `data/app/`. To rebuild it from scratch, download both datasets into `data/raw/` and run:

```bash
.venv/bin/python -m pipeline.run                 # about 2 minutes
.venv/bin/python -m tests.test_golden_sellers    # independent check
```

Datasets: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and [Marketing Funnel by Olist](https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist).

## Deploy (Streamlit Community Cloud, free)

1. Push this folder to a public GitHub repository. `.gitignore` already excludes the environment and raw data.
2. Sign in at share.streamlit.io with the same GitHub account, choose **Create app**, pick the repository, and set the main file to `app/streamlit_app.py`.
3. The public link looks like `your-app-name.streamlit.app`. Free apps sleep after a few days without visitors; open the link once before presenting.

## Data rules

* Complete months only: Jan 2017 to Aug 2018.
* An order with several sellers (1.3%) is credited to the seller with the highest item value; a seller's first sale counts any order containing their item.
* One review per order (the latest answered).
* Dispatch deadline is the latest `shipping_limit_date` in the order. Late dispatch compares it with the carrier handover time, so the seller is never blamed for carrier delays.
* Late delivery: delivered on a later calendar day than the promised date, or handed to the carrier and never delivered. Late dispatch includes orders never handed over after the deadline passed.
* Training and evaluation use orders promised by 29 Aug 2018 (the clean cohort).
* Lead numbers use leads first contacted from Dec 2017, because the deals file only holds wins from then on.
* Funnel stages are strictly conditional: each stage counts only orders that passed every earlier stage.

## Limitations

* The data has no interventions, so the platform shows when acting is possible, not that acting works.
* 62% of seller exits show no prior decline; the reason a seller leaves is not recorded.
* Cancelled orders are left out of live queues because cancellation time is not recorded.
* Complaint tags come from Portuguese keyword rules checked by reading 200 reviews: precise (80% to 100% per tag) but they miss some complaints, so shares are lower bounds.
* Model choice: scikit learn gradient boosting (same algorithm family as LightGBM, no system dependency).
