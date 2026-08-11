"""miner_1 2026-09-10: Library correlation audit for trend_tstat_20 (row-aligned).

Library .signal.npy artifacts are row-aligned to the price calendar: row i of an
artifact with shape (N,15) corresponds to calendar row i of the full price panel.
No provenance JSON required. Compare fac.iloc[:N] vs artifact.
"""
import numpy as np
import pandas as pd
import glob, os
import sys
sys.path.insert(0, 'scripts')
from miner_1_20260910_utils import load_panel, align_close

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
print(f'candidate shape {fac.shape} {fac.index[0].date()}..{fac.index[-1].date()}')

lib_corrs = {}
for npy in sorted(glob.glob(os.path.join('factors', '*.signal.npy'))):
    fname = os.path.basename(npy)
    fid = fname.replace('.signal.npy', '')
    arr = np.load(npy)
    N = arr.shape[0]
    if arr.shape[1] != 15 or N < 1000 or N > fac.shape[0]:
        print(f'lib {fid:35s} SKIP shape={arr.shape}')
        continue
    lib = pd.DataFrame(arr, index=fac.index[:N], columns=fac.columns)
    a = fac.iloc[:N]
    corrs = []
    for c in close.columns:
        x = a[c].astype(float); y = lib[c].astype(float)
        m = x.notna() & y.notna()
        if m.sum() >= 60:
            r = np.corrcoef(x[m], y[m])[0, 1]
            if np.isfinite(r):
                corrs.append(r)
    if corrs:
        maxabs = max(abs(r) for r in corrs)
        meanabs = float(np.mean([abs(r) for r in corrs]))
        lib_corrs[fid] = (maxabs, meanabs, len(corrs))
        print(f'lib {fid:35s} shape={N:5d} maxabs_corr={maxabs:.3f} meanabs={meanabs:.3f} n_pairs={len(corrs)}')

if lib_corrs:
    overall_max = max(v[0] for v in lib_corrs.values())
    print(f'\nMAX_ABS_LIBRARY_CORRELATION = {overall_max:.4f}')
    for fid, (mx, mn, np_) in sorted(lib_corrs.items(), key=lambda kv: -kv[1][0])[:12]:
        print(f'  {fid:35s} {mx:.3f}')
else:
    print('\nNO LIBRARY CORRELATIONS COMPUTED')
print('\nDONE')
