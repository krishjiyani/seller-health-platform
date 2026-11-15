"""Live seller signals for the dispatch window: what a KAM could see about a seller on the morning of a given day.

Every value uses only events that happened before that day started, so a snapshot scored on day D never sees D itself.
Measured gain (feasibility test F3 in analysis/iter2): dispatch window AUC 0.768 to 0.787, better in 12 of 12 test months."""
import numpy as np
import pandas as pd

LIVE_SELLER = ["s_open_now", "s_late_14", "s_hand_n_14", "s_surge", "s_handover_med_30"]


def seller_day_stats(o: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per (seller, day).

    s_open_now: orders bought in the last 30 days and not yet handed over (includes orders later cancelled, because their
                fate is unknown live).
    s_late_14: late share among handovers completed in the last 14 days (NaN when there were none).
    s_hand_n_14: number of those handovers.
    s_surge: orders bought in the last 7 days over the seller's 7 day average across the 90 days before that.
    s_handover_med_30: median days from purchase to handover, over handovers completed in the last 30 days."""
    x = o[o.seller_id.notna()][["seller_id", "ts_purchase", "ts_carrier", "late_dispatch"]].copy()
    x["handover_d"] = (x.ts_carrier - x.ts_purchase).dt.total_seconds() / 86400
    sid = x.seller_id.astype("category")
    x["s"] = sid.cat.codes
    d0 = days.min() - pd.Timedelta(days=120)
    nd = (days.max() - d0).days + 2
    pday = ((x.ts_purchase.dt.normalize() - d0).dt.days).clip(0, nd - 1).values
    hday = (x.ts_carrier.dt.normalize() - d0).dt.days
    S = len(sid.cat.categories)

    def cumulative(day_idx, weights=None, mask=None):
        g = np.zeros((S, nd))
        m = np.ones(len(x), bool) if mask is None else mask
        np.add.at(g, (x.s.values[m], day_idx[m]), 1 if weights is None else weights[m])
        return g.cumsum(1)

    purch = cumulative(pday)
    handed = hday.notna().values & (hday.values >= 0)
    hidx = np.where(handed, hday.fillna(0).astype(int).clip(0, nd - 1).values, 0)
    hand = cumulative(hidx, mask=handed)
    late = cumulative(hidx, weights=x.late_dispatch.fillna(0).values, mask=handed)
    # Open orders: +1 on the purchase day, -1 on the handover day or 30 days after purchase, whichever comes first.
    end = np.where(handed, np.minimum(hidx, pday + 30), pday + 30).clip(0, nd - 1)
    diff = np.zeros((S, nd + 1))
    np.add.at(diff, (x.s.values, pday), 1)
    np.add.at(diff, (x.s.values, end), -1)
    open_now = diff.cumsum(1)[:, :nd]
    hsorted = x[handed].sort_values("ts_carrier")
    out = []
    for day in days:
        j = (day - d0).days            # column of this day; cumulative counts at j - 1 cover events before the day
        f = pd.DataFrame({"seller_id": sid.cat.categories})
        f["day"] = day
        f["s_open_now"] = open_now[:, j - 1]
        n14 = hand[:, j - 1] - hand[:, j - 15]
        f["s_late_14"] = np.where(n14 > 0, (late[:, j - 1] - late[:, j - 15]) / np.maximum(n14, 1), np.nan)
        f["s_hand_n_14"] = n14
        o7 = purch[:, j - 1] - purch[:, j - 8]
        o90 = purch[:, j - 8] - purch[:, j - 98]
        f["s_surge"] = (o7 + 1) / (o90 / 13 + 1)
        w = hsorted[(hsorted.ts_carrier < day) & (hsorted.ts_carrier >= day - pd.Timedelta(days=30))]
        f["s_handover_med_30"] = f.seller_id.map(w.groupby("seller_id").handover_d.median())
        out.append(f)
    stats = pd.concat(out, ignore_index=True)
    stats["seller_id"] = stats.seller_id.astype(str)
    return stats
