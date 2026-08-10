"""Calibration: reproduce ret_skew_10 IC and gate-style pairwise correlations
to verify the evaluation framework matches the deterministic post-Miner gate."""
import json, sys
import numpy as np
sys.path.insert(0, 'scripts')
from miner_1_20260813_common import (load_assets, load_library_signals,
                                     daily_rank_ic_matrix, ic_stats,
                                     mean_daily_spearman, zscore_winsor)

closes, ohlcv, grid, vt = load_assets()
P = closes.values.astype(float)          # (2408, 15) close price matrix
n = P.shape[0]

# forward 10d return (paper: known at t, using closes through t+10)
fwd10 = np.full_like(P, np.nan)
for h in [10]:
    fwd10[:-h, :] = P[h:, :] / P[:-h, :] - 1.0

# ret_skew_10 factor: rolling skewness of 10d daily returns
ret = np.full_like(P, np.nan)
ret[1:, :] = P[1:, :] / P[:-1, :] - 1.0

def rolling_skew(r, w, minp):
    out = np.full_like(r, np.nan)
    for t in range(w - 1, r.shape[0]):
        win = r[t - w + 1:t + 1, :]
        m = ~np.isnan(win)
        cnt = m.sum(axis=0)
        ok = cnt >= minp
        if ok.any():
            mu = np.nanmean(win, axis=0)
            sd = np.nanstd(win, axis=0, ddof=1)
            s = np.where(sd > 0, np.nanmean(((win - mu) / sd) ** 3, axis=0), np.nan)
            out[t, ok] = s[ok]
    return out

skew10 = rolling_skew(ret, 10, 5)
ics = daily_rank_ic_matrix(skew10, fwd10)
st = ic_stats(ics)
print('ret_skew_10 recomputed:', {k: round(v, 4) if isinstance(v, float) else v for k, v in st.items()})
print('expected ~ ic 0.0311 icir 0.1003')

# gate-style correlation sweep vs library artifacts
lib = load_library_signals()
print('library artifacts loaded:', len(lib))
raw = skew10
z = zscore_winsor(raw)
for label, sig in [('RAW', raw), ('RANKZ', z)]:
    print(f'--- correlation sweep using {label} ---')
    res = []
    for fid, d in lib.items():
        r = mean_daily_spearman(sig, d['matrix'])
        if not np.isnan(r):
            res.append((abs(r), fid, r))
    res.sort(reverse=True)
    for a, fid, r in res[:8]:
        print(f'  {fid:28s} abs_rho={a:.4f} rho={r:.4f}')
