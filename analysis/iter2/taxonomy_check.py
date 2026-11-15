"""Precision check for the complaint taxonomy (pipeline/taxonomy.py).

Draws 25 random 1 to 2 star reviews per tag, plus 50 with text that match no tag, and writes them to
out/taxonomy_sample.csv for reading. The reader's verdicts go in out/taxonomy_check.csv (columns: sample_id, correct),
and `--score` prints precision per tag.
Run: python -m analysis.iter2.taxonomy_check [--score]
Note: without --score this redraws the sample with the current rules and overwrites out/taxonomy_sample.csv."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from analysis.iter2.base import load
from pipeline.taxonomy import TAGS, tag_reviews

OUT = Path(__file__).parent / "out"
o, _ = load()
low = o[(o.low_review == 1) & o.review_text.notna() & (o.review_text.str.strip().str.len() > 3)].copy()
tags = tag_reviews(low.review_text)
rows = []
for k in TAGS:
    s = low[tags[k]].sample(25, random_state=7)
    rows += [dict(tag=k, order_id=r.order_id, text=r.review_text.replace("\n", " ")) for r in s.itertuples()]
none = low[~tags.any(axis=1)].sample(50, random_state=7)
rows += [dict(tag="(no tag)", order_id=r.order_id, text=r.review_text.replace("\n", " ")) for r in none.itertuples()]
sample = pd.DataFrame(rows)
sample.insert(0, "sample_id", range(len(sample)))

if "--score" in sys.argv:
    # Score the saved sample (drawn 2026-11-12 with the first version of the rules, before tags were renamed).
    lab = pd.read_csv(OUT / "taxonomy_check.csv")
    m = pd.read_csv(OUT / "taxonomy_sample.csv").drop(columns="tag").merge(lab, on="sample_id")
    print("precision per tag (share of matched reviews where the tag is right):")
    print(m[m.tag != "(no tag)"].groupby("tag").correct.agg(["mean", "size"]).round(2).to_string())
    print(f"untagged reviews that should have had one of the tags: {1 - m[m.tag == '(no tag)'].correct.mean():.0%} of 50")
else:
    sample.to_csv(OUT / "taxonomy_sample.csv", index=False)
    pd.set_option("display.max_colwidth", 220)
    for r in sample.itertuples():
        print(f"{r.sample_id:3d} [{r.tag}] {r.text[:220]}")
