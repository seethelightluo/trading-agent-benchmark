"""Check dd_duration candidates vs ALL effective root library factors (artifact-based rho)."""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, factor_to_panel, canonical_grid, signal_matrix,
                           WATCHLIST, forward_returns, rank_ic_series, VAL_START, VAL_END)

prices = load_prices(days=2100)
grid = canonical_grid(prices)


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
    return panel.sub(panel.mean(axis=1), axis=0).div(panel.std(axis=1), axis=0)


def orthogonalize(panel, ref, min_valid=8):
    z, zr = zscore_rows(panel), zscore_rows(ref)
    out = z.copy()
    for d in z.index:
        if d not in zr.index:
            out.loc[d] = np.nan
            continue
        x, y = z.loc[d], zr.loc[d]
        m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if m.sum() < min_valid:
            out.loc[d] = np.nan
            continue
        xv, yv = x[m].values, y[m].values
        xv = (xv - xv.mean()) / (xv.std() + 1e-12)
        yv = (yv - yv.mean()) / (yv.std() + 1e-12)
        beta = float(np.dot(xv, yv) / (len(xv) + 1e-12))
        out.loc[d, m] = xv - beta * yv
    return out


def effective_library():
    mats = {}
    for jp in sorted(Path('factors').glob('*.json')):
        try:
            payload = json.loads(jp.read_text())
        except Exception:
            continue
        if payload.get('validation', {}).get('status') != 'EFFECTIVE':
            continue
        art = payload.get('signal_artifact')
        if not art:
            continue
        p = jp.parent / str(art)
        if not p.exists():
            continue
        mats[payload['factor_id']] = np.load(p, allow_pickle=False)
    return mats


lib = effective_library()
print('effective library factors (%d):' % len(lib), sorted(lib))
# artifacts were saved on an earlier (shorter) canonical grid -> they are the tail of current grid
lib_df = {}
for k, v in lib.items():
    n = len(v)
    idx = grid[-n:] if n <= len(grid) else grid
    lib_df[k] = pd.DataFrame(v[-min(n, len(grid)):], index=idx, columns=WATCHLIST)


def pair_rho(a, b, min_valid=8):
    corrs = []
    for i in range(len(a)):
        x, y = a[i], b[i]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= min_valid:
            r = pd.Series(x[m]).rank().corr(pd.Series(y[m]).rank())
            if np.isfinite(r):
                corrs.append(r)
    return float(np.mean(corrs)) if corrs else np.nan


def ic10(panel):
    fwd = forward_returns(prices, 10)
    s = rank_ic_series(panel, fwd)
    s = s[(s.index >= VAL_START) & (s.index <= VAL_END)]
    return (float(s.mean()),
            float(s.mean() / s.std(ddof=1)) if len(s) > 1 and s.std(ddof=1) > 0 else 0.0)


p120 = factor_to_panel(lambda df, s: dd_duration(df, s, 120), prices)
p60 = factor_to_panel(lambda df, s: dd_duration(df, s, 60, 30), prices)
mom_ref = lib_df.get('mom_120d_skip5')
p120r = orthogonalize(p120, mom_ref if mom_ref is not None else p120 * 0)

for fid, panel in [('dd_dur_120', p120), ('dd_dur_60', p60), ('dd_dur_120_resid', p120r)]:
    mat = signal_matrix(panel, grid)
    rhos = {lid: pair_rho(mat, lm) for lid, lm in lib.items()}
    best = max(rhos, key=lambda k: abs(rhos[k]))
    ic, icir = ic10(panel)
    over = ', '.join(f'{k}={v:.2f}' for k, v in rhos.items() if abs(v) > 0.4)
    print(f'{fid}: IC10={ic:.4f} ICIR10={icir:.4f} max|rho|={abs(rhos[best]):.3f} vs {best} | >0.4: {over}')
