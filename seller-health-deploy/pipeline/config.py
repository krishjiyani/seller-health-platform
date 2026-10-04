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

# Probability cutoffs for the "high risk" flag (roughly 3x each base rate).
HIGH_DISPATCH = 0.25
HIGH_CARRIER = 0.30
HIGH_SILENCE = 0.50
