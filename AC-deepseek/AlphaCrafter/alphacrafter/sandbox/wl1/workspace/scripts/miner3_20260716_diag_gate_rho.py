"""miner_3 diagnostic: replicate the post-miner gate's pooled-rho logic.

Hypothesis: the gate's pairwise abs_spearman_rho is computed on pooled
(date x symbol) values, so any signal with persistent cross-sectional
structure (vol levels, average momentum) shows high rho vs mom_10d_skip5.
Test whether per-symbol time-series standardization / cross-sectional
rank-then-z-score reduces rho below the 0.5 correlation threshold.
"""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close

closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2020-01-01")) & (idx <= pd.Timestamp("2026-07-15"))]
print("full idx len:", len(idx))

CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
OP = pd.DataFrame({s: closes[s]["open"].reindex(idx).astype(float) for s in SYMBOLS})

mom = np.load("factors/miner2_20260716_mom_10d_skip5.npy", allow_pickle=True)
print("mom npy shape:", mom.shape)

rev = (1.0 - CP / OP).to_numpy(dtype=float)
LRET = np.log(CP / CP.shift(1))
vol20 = LRET.rolling(20).std().to_numpy(dtype=float)


def pooled_rho(a, b):
    m = (~np.isnan(a)) & (~np.isnan(b))
    if m.sum() < 500:
        return float("nan")
    return float(spearmanr(a[m], b[m]).statistic)


def zscore_cols(X):
    Z = np.full_like(X, np.nan, dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        mu = np.nanmean(col)
        sd = np.nanstd(col)
        if sd > 0:
            Z[:, j] = (col - mu) / sd
    return Z


def csrank_then_z(X):
    df = pd.DataFrame(X, index=idx, columns=SYMBOLS)
    rk = df.rank(axis=1, pct=True)
    return zscore_cols(rk.to_numpy(dtype=float))


print("\npooled rho vs mom_10d_skip5 artifact:")
print("  rev_intra raw        :", round(pooled_rho(rev, mom), 4))
print("  rev_intra zscored    :", round(pooled_rho(zscore_cols(rev), mom), 4))
print("  vol20 raw            :", round(pooled_rho(vol20, mom), 4))
print("  vol20 zscored        :", round(pooled_rho(zscore_cols(vol20), mom), 4))
print("  rev cs-rank-z        :", round(pooled_rho(csrank_then_z(rev), mom), 4))
print("  vol20 cs-rank-z      :", round(pooled_rho(csrank_then_z(vol20), mom), 4))

momsig = (np.log(CP / CP.shift(10)) - np.log(CP / CP.shift(5))).to_numpy(dtype=float)
print("\nmom npy vs real mom10skip5 pooled rho:", round(pooled_rho(mom, momsig), 4))
print("mom npy col means:", np.round(np.nanmean(mom, axis=0), 4))
print("mom npy col stds :", np.round(np.nanstd(mom, axis=0), 4))
