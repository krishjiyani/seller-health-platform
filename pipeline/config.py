"""Paths, analysis windows and fixed parameters shared by every pipeline step."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
APP = ROOT / "data" / "app"

# Complete order months only (2016 is sparse; Sep and Oct 2018 hold 20 orders in total).
DATA_START = pd.Timestamp("2017-01-01")
DATA_END = pd.Timestamp("2018-08-31 23:59:59")

# First month that gets out of sample predictions (earlier months only train).
SCORE_FROM = pd.Period("2017-04", "M")
SILENCE_SCORE_FROM = pd.Period("2017-08", "M")

SEED = 42
GBM_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, random_state=SEED)
SILENCE_PARAMS = dict(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=40, random_state=SEED)

# Shrinkage strength for per seller rates: (C * prior + successes) / (C + n).
SHRINK_C = 10

# Silence: probability cutoff for "Act now" / "Window open". Silence scores are close to calibrated, so a fixed cutoff works.
HIGH_SILENCE = 0.50
# Order windows show risk tiers by daily rank instead of percentages: order probabilities drift with the promise policy and
# with carrier shocks (audit S5, S7 in analysis/iter2), while the ranking holds. High = top 10% of the day's open orders.
TIER_HIGH, TIER_ELEVATED = 0.10, 0.30
# Used only by the window start reference models in metrics.json.
HIGH_DISPATCH = 0.25
HIGH_CARRIER = 0.30

# Data rules (evidence in analysis/iter2/SCOPE_ITERATION2.md).
# The public data only keeps orders with an answered review, and reviews stop on 31 Aug 2018, so orders promised after
# 29 Aug 2018 are a fast delivery only sample. Training and evaluation use orders promised on or before this date.
CLEAN_PROMISE_MAX = pd.Timestamp("2018-08-29")
# The review survey is created the day after delivery, or 2 days after the promised date if the parcel has not arrived
# (exact for 92.7% of delivered orders). So the carrier window closes at promise + 2 days.
SURVEY_AFTER_PROMISE_D = 2
# A parcel still with the carrier this many days after handover is flagged as possibly lost.
LOST_FLAG_D = 20
# Sellers this big (orders in the last 90 days) are ranked by days since last order: the model lost to recency for them.
BIG_SELLER_O90 = 10
# Lead funnel: the deals file only holds wins from Dec 2017, so lead numbers use MQLs first contacted from then on.
LEAD_COHORT_FROM = pd.Timestamp("2017-12-01")
