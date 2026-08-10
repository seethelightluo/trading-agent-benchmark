"""dd_duration variants + orthogonalization vs mom_120d_skip5.

Goal: keep predictive power of drawdown-duration while pushing
max_abs_library_correlation below 0.5 so the deterministic gate keeps it.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           build_library_panels, max_library_correlation)

prices = load_prices(days=2100)
lib = build_library_panels(prices)
mom120 = lib['mom_120d_skip5']


def dd_duration(df, s, win=120, minp=60):
    c = df['close']
    h = c.rolling(win, min_periods=minp).max()
    is_high = (c >= h).fillna(False)
    idx_high = np.flatnonzero(is_high.values)
    pos = np.arange(len(c))
    last = np.searchsorted(idx_high, pos) - 1
    dur = np.where(last >= 0, pos - idx_high[np.maximum(last, 0)], np.nan)
    return pd.Series(np.log1p(dur), index=c.index)


def zscore_rows(panel):
    mu = panel.mean(axis=1)
    sd = panel.std(axis=1)
    return panel.sub(mu, axis=0).div(sd, axis=0)


def orthogonalize(panel, ref, min_valid=8):
    """Per-date cross-sectional OLS residual of z(panel) on z(ref)."""
    z = zscore_rows(panel)
    zr = zscore_rows(ref)
    out = z.copy()
    for d in z.index:
        if d not in zr.index:
            out.loc[d] = np.nan
            continue
        x = z.loc[d]
        y = zr.loc[d]
        m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if m.sum() < min_valid:
            out.loc[d] = np.nan
            continue
        xv, yv = x[m].values, y[m].values
        xv = (xv - xv.mean()) / (xv.std() + 1e-12)
        yv = (yv - yv.mean()) / (yv.std() + 1e-12)
        beta = float(np.dot(xv, yv) / (len(xv) + 1e-12))  # corr approx
        resid = xv - beta * yv
        out.loc[d, m] = resid
    return out


cands = {
    'dd_dur_120': lambda df, s: dd_duration(df, s, 120),
    'dd_dur_60': lambda df, s: dd_duration(df, s, 60, 30),
    'dd_dur_250': lambda df, s: dd_duration(df, s, 250, 120),
}

for fid, fn in cands.items():
    p = factor_to_panel(fn, prices)
    m = validate_factor(fid, p, prices)
    rho, rid = max_library_correlation(p, lib)
    print(f'{fid}: IC={m["ic"]:.4f} ICIR={m["icir"]:.4f} hit={m["ic_hit_ratio"]:.3f} '
          f'rho={rho:.3f}({rid}) PASS={abs(m["ic"])>=0.007 and abs(m["icir"])>=0.084}')

# orthogonalized versions
for fid, fn in cands.items():
    p = factor_to_panel(fn, prices)
    pr = orthogonalize(p, mom120)
    m = validate_factor(fid + '_resid', pr, prices)
    rho, rid = max_library_correlation(pr, lib)
    print(f'{fid}_resid: IC={m["ic"]:.4f} ICIR={m["icir"]:.4f} hit={m["ic_hit_ratio"]:.3f} '
          f'rho={rho:.3f}({rid}) PASS={abs(m["ic"])>=0.007 and abs(m["icir"])>=0.084}')
