"""miner_1 2026-09-10: Full validation of trend_tstat_20 candidate.

1. Recompute factor, IC/ICIR stats + decay profile
2. Correlation vs existing library signal artifacts (pairwise, on overlapping dates)
3. Persist-ready metrics
"""
import numpy as np
import pandas as pd
import json, glob, os
import sys
sys.path.insert(0, 'scripts')
from miner_1_20260910_utils import (load_panel, align_close, forward_returns,
                                    daily_ic, summarize_ic, turnover_rank, coverage, decay_profile)

panel = load_panel(days=2500)
close = align_close(panel)
logp = np.log(close)
n = len(logp)
t = pd.Series(np.arange(n), index=logp.index, dtype=float)
WINDOW = 20

def trend_tstat(logp, window):
    def _col_ts(x):
        w = max(10, window // 2)
        var_t = t.rolling(window, min_periods=w).var()
        cov = x.rolling(window, min_periods=w).cov(t)
        var_y = x.rolling(window, min_periods=w).var()
        b = cov / var_t
        r2 = (cov ** 2) / (var_y * var_t)
        n_eff = x.rolling(window, min_periods=w).count()
        ss_res = var_y * (n_eff - 1) * (1 - r2)
        sxx = var_t * (n_eff - 1)
        se = np.sqrt((ss_res / ((n_eff - 2) * sxx)).clip(lower=1e-12))
        return b / se
    return logp.apply(_col_ts)

fac = trend_tstat(logp, WINDOW)
fwd10 = forward_returns(close, 10)
ics = daily_ic(fac, fwd10, min_assets=8)
s = summarize_ic(ics, f'trend_tstat_{WINDOW}')
cov, d8 = coverage(fac, close)
print(f'coverage={cov:.3f} dates_ge8={d8:.3f} turnover10={turnover_rank(fac):.3f}')
dec = decay_profile(fac, close, max_h=20)
print('decay IC by horizon:', {k: round(v, 4) for k, v in dec.items()})

# ---- correlation vs existing library artifacts ----
lib_corrs = {}
facs_dir = 'factors'
for npy in sorted(glob.glob(os.path.join(facs_dir, '*.signal.npy'))):
    fname = os.path.basename(npy)
    fid = fname.replace('.signal.npy', '')
    # find provenance in matching json
    jpath = os.path.join(facs_dir, fid + '.json')
    prov = None
    if os.path.exists(jpath):
        try:
            with open(jpath) as f:
                prov = json.load(f).get('artifact_provenance', {})
        except Exception:
            prov = None
    if prov is None:
        continue
    dates_first = prov.get('dates_first')
    dates_last = prov.get('dates_last')
    arr = np.load(npy)
    if arr.shape[1] != 15:
        continue
    # build DataFrame on library index (union dates). Use provenance dates.
    try:
        lib_idx = pd.date_range(dates_first, dates_last, periods=arr.shape[0])
    except Exception:
        continue
    lib = pd.DataFrame(arr, index=lib_idx, columns=close.columns)
    # align to my factor dates: take common index subset (first 2398 rows approx)
    common = fac.index.intersection(lib.index)
    if len(common) < 100:
        continue
    a = fac.loc[common]
    b = lib.loc[common]
    corrs = []
    for c in close.columns:
        x = a[c].astype(float); y = b[c].astype(float)
        m = x.notna() & y.notna()
        if m.sum() >= 60:
            r = np.corrcoef(x[m], y[m])[0, 1]
            if np.isfinite(r):
                corrs.append(r)
    if corrs:
        maxabs = max(abs(r) for r in corrs)
        meanabs = np.mean([abs(r) for r in corrs])
        lib_corrs[fid] = (maxabs, meanabs, len(corrs))
        print(f'lib {fid:35s} maxabs_corr={maxabs:.3f} meanabs={meanabs:.3f} n_pairs={len(corrs)}')

if lib_corrs:
    overall_max = max(v[0] for v in lib_corrs.values())
    print(f'\nMAX_ABS_LIBRARY_CORRELATION = {overall_max:.4f}')
    for fid, (mx, mn, np_) in sorted(lib_corrs.items(), key=lambda kv: -kv[1][0])[:10]:
        print(f'  {fid:35s} {mx:.3f}')

# save candidate panel for persistence
np.save('scripts/trend_tstat_20.candidate.npy', fac.values)
fac.to_csv('scripts/trend_tstat_20.candidate.csv')
print('\nDONE')
