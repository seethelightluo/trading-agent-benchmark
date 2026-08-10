"""miner_3 - pairwise correlation among gate-passing candidates (2026-07-16).
Goal: select a diverse subset (pairwise rho < 0.5) for persistence so the
deterministic post-Miner pairwise gate does not reject duplicates.
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close
import miner3_fast as F

closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2020-01-01"))]
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
HP = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in SYMBOLS})
LP = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in SYMBOLS})
RET = CP.pct_change()
vol20 = RET.rolling(20).std() * np.sqrt(252)

panels = {}
panels["rev_1d"] = -(CP / CP.shift(1) - 1.0)
panels["rev_2d"] = -(CP / CP.shift(2) - 1.0)
panels["rev_3d"] = -(CP / CP.shift(3) - 1.0)
panels["rev_5d"] = -(CP / CP.shift(5) - 1.0)
panels["rev_3d_vol"] = -(CP / CP.shift(3) - 1.0) / vol20
panels["clv_1d"] = (CP - LP) / (HP - LP + 1e-12)
panels["clv_5d"] = (CP - LP.rolling(5).min()) / (HP.rolling(5).max() - LP.rolling(5).min() + 1e-12)

names = list(panels.keys())
# pooled Pearson rho over all valid (date, symbol) cells after per-asset z-scoring
pooled = pd.DataFrame({n: panels[n].stack() for n in names})
pooled = pooled.dropna()
print("pooled valid cells:", len(pooled))
# z-score each column to remove level differences
z = (pooled - pooled.mean()) / pooled.std()
rho_pooled = z.corr()
print("\n=== pooled Pearson rho ===")
print(rho_pooled.round(3))

# per-date rank corr averaged (time-series of cross-sectional Spearman)
def avg_ts_rankcorr(a, b):
    vals = []
    for dt in idx:
        x = panels[a].loc[dt].dropna()
        y = panels[b].loc[dt].dropna()
        common = x.index.intersection(y.index)
        if len(common) >= 8:
            vals.append(x[common].rank().corr(y[common].rank()))
    return float(np.nanmean(vals)) if vals else np.nan

print("\n=== avg per-date rank corr (Spearman) ===")
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        r = avg_ts_rankcorr(names[i], names[j])
        print(f"{names[i]:12s} vs {names[j]:12s}: {r:+.3f}")
