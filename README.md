# Seller Health and Intervention Platform

A decision support console for key account managers (KAMs) on an ecommerce marketplace. It predicts **when** to intervene, at order level and seller level, while there is still time to prevent the damage, and recommends the action. Built on the public Olist Brazilian marketplace data (Jan 2017 to Aug 2018).

Final year BTech project, Group BCAIAA61, MIT School of Computing, MIT ADT University, Pune. Guide: Prof. Sneha Singha.

## What it predicts

| Window | Question | Time to act | Model AUC | Best simple rule |
|---|---|---|---|---|
| Dispatch | Will this order miss the seller's dispatch deadline? | about 6 days | 0.76 | 0.72 (days waiting) |
| Carrier | Will this parcel arrive after the promised date? | about 20 days | 0.79 | 0.77 (days left) |
| Silence | Will this seller stop selling within 1 to 3 months? | 1 to 3 months | 0.82 to 0.79 | 0.79 to 0.76 (recency) |
| Onboarding | Will this signed seller ever start selling? | day 0 to 44 | measured decay curve | |

AUC is averaged over monthly walk forward tests from Sep 2017 to Aug 2018: every month is scored by a model trained only on outcomes known before it. Order risks are re-scored every day an order stays inside its window. Full results, including what the data cannot predict, are on the app's Evidence page and in `analysis/PREDICTABILITY_AUDIT.md`.

## Project layout

```
app/                 Streamlit app (entry: app/streamlit_app.py, pages in app/views/)
pipeline/            raw CSVs to app tables: build.py (tables + assertions), models.py (walk forward models), run.py
tests/               golden seller check: recomputes sellers from raw CSVs by an independent route
analysis/            one off predictability audit scripts and their write up
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
.venv/bin/python -m pipeline.run                 # about 3 minutes
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
* Funnel stages are strictly conditional: each stage counts only orders that passed every earlier stage.

## Limitations

* The data has no interventions, so the platform shows when acting is possible, not that acting works.
* 62% of seller exits show no prior decline; the reason a seller leaves is not recorded.
* Cancelled orders are left out of live queues because cancellation time is not recorded.
* Model choice: scikit learn gradient boosting (same algorithm family as LightGBM, no system dependency).
