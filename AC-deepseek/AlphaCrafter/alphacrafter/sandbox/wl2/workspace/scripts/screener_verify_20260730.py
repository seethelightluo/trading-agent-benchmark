"""Screener verification: 2026-07-30 cycle.
Loads the 3 active signal artifacts, applies the as-consumed transform
(neutral-fill -> CS rank -> z-score -> winsorize 3sigma), and verifies
pairwise correlations and quality-IC-tilt weights.
"""
import json
import numpy as np
from pathlib import Path

ACTIVE = ["mom20_volproxy60", "dxy_beta_cond_60x20", "calmness_20"]
BASE = Path("factors")

def load_signal(fid):
    arr = np.load(BASE / f"{fid}.signal.npy")
    return arr

def transform(x):
    """as-consumed transform: neutral-fill -> CS rank -> z-score -> winsorize 3sigma"""
    out = np.full_like(x, np.nan, dtype=float)
    for t in range(x.shape[0]):
        row = x[t]
        m = ~np.isnan(row)
        if m.sum() == 0:
            continue
        # neutral fill: median for NaN
        filled = row.copy()
        filled[~m] = np.nanmedian(row[m])
        # CS rank -> percentile
        ranks = np.argsort(np.argsort(filled))
        pct = ranks / (len(filled) - 1) if len(filled) > 1 else 0.5
        # z-score of percentile
        z = (pct - 0.5) * np.sqrt(12.0)  # approx std of uniform
        # winsorize at 3 sigma
        z = np.clip(z, -3.0, 3.0)
        out[t] = z
    return out

def spearman(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 5:
        return np.nan
    ra = np.argsort(np.argsort(a[m]))
    rb = np.argsort(np.argsort(b[m]))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else np.nan

# load metrics from JSONs
metrics = {}
for fid in ACTIVE:
    with open(BASE / f"{fid}.json") as f:
        j = json.load(f)
    met = j["validation"]["metrics"]
    metrics[fid] = {
        "ic": met["ic"], "icir": met["icir"],
        "q": abs(met["ic"]) * abs(met["icir"]),
        "turnover_10d_rank": met["turnover_10d_rank"],
        "coverage_asset_days": met["coverage_asset_days"],
        "max_abs_library_corr": met["max_abs_library_correlation"],
    }

print("=== QUALITY (q=|IC|*|ICIR|) ===")
qsum = sum(m["q"] for m in metrics.values())
for fid, m in metrics.items():
    w = m["q"] / qsum
    print(f"{fid:24s} IC={m['ic']:.4f} ICIR={m['icir']:.4f} q={m['q']:.6f} w={w:.6f} "
          f"turn={m['turnover_10d_rank']:.3f} cov={m['coverage_asset_days']:.3f} "
          f"maxLibCorr={m['max_abs_library_corr']:.4f}")
print(f"weights_sum={sum(m['q']/qsum for m in metrics.values()):.6f}")

print("\n=== SIGNAL ARTIFACT SHAPES ===")
signals = {fid: load_signal(fid) for fid in ACTIVE}
for fid, s in signals.items():
    print(f"{fid:24s} shape={s.shape} nan={np.isnan(s).sum()}")

print("\n=== AS-CONSUMED TRANSFORMED PAIRWISE SPEARMAN ===")
tr = {fid: transform(s) for fid, s in signals.items()}
fids = ACTIVE
for i in range(len(fids)):
    for j in range(i + 1, len(fids)):
        a, b = tr[fids[i]], tr[fids[j]]
        full = spearman(a, b)
        last250 = spearman(a[-250:], b[-250:])
        last60 = spearman(a[-60:], b[-60:])
        flag = "OK" if abs(full) < 0.5 and abs(last250) < 0.5 and abs(last60) < 0.5 else "CHECK"
        print(f"{fids[i]:24s} vs {fids[j]:24s} full={full:+.4f} last250={last250:+.4f} "
              f"last60={last60:+.4f}  [{flag}]")

# last available row sanity: how many non-nan per asset on final date
print("\n=== LAST ROW NON-NAN COUNT (per factor, final date 2026-07-29) ===")
for fid in ACTIVE:
    s = signals[fid]
    print(f"{fid:24s} last-row non-nan = {np.sum(~np.isnan(s[-1]))}/15")
