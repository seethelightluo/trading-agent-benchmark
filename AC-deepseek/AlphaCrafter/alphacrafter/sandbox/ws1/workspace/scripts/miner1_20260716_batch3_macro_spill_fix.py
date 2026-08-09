"""miner_1 batch-3a fix: cross-asset macro-spillover beta factors.

References other watchlist benchmarks (WTI/COPPER/XAU/US10Y/CN10Y/SPX) from the
close panel, and observation-only macros (USDJPY/VIX/DXY) from macro dict.
factor = sign * rolling_beta(asset_ret, bench_ret, 60) * bench_mom(20).
"""
import numpy as np
import pandas as pd
from miner1_20260716_lib import (build_panel, factor_values, forward_returns,
                                 daily_ic, summarize)

H = 10
MIN_VALID = 8


def bench_beta_cond(bench_key, sign=1.0, win=60, mom=20):
    """bench_key in panel['macro'] or panel['closes']."""
    def fn(sym, close, volume, panel=None):
        if panel is None:
            return None
        grid = panel['grid']
        if bench_key in panel['macro']:
            bench = panel['macro'][bench_key].reindex(grid)
        else:
            bench = panel['closes'][bench_key].reindex(grid)
        if bench is None or bench.dropna().empty or sym == bench_key:
            return None
        r_a = close.pct_change().reindex(grid)
        r_m = bench.pct_change()
        beta = r_a.rolling(win, min_periods=30).cov(r_m) / r_m.rolling(win, min_periods=30).var()
        mm = (bench / bench.shift(mom) - 1.0)
        return (sign * beta * mm).replace([np.inf, -np.inf], np.nan)
    return fn


CANDIDATES = [
    ('usdjpy_beta_cond_60x20', bench_beta_cond('USDJPY', sign=1.0)),
    ('wti_beta_cond_60x20', bench_beta_cond('WTI', sign=1.0)),
    ('copper_beta_cond_60x20', bench_beta_cond('COPPER', sign=1.0)),
    ('xau_beta_cond_60x20', bench_beta_cond('XAU', sign=1.0)),
    ('us10y_beta_cond_60x20', bench_beta_cond('US10Y', sign=1.0)),
    ('cn10y_beta_cond_60x20', bench_beta_cond('CN10Y', sign=1.0)),
    ('spx_beta_cond_60x20', bench_beta_cond('SPX', sign=1.0)),
    ('dxy_beta_cond_60x20', bench_beta_cond('DXY', sign=-1.0)),
]


def with_panel(fn, panel):
    def wrapped(sym, close, volume):
        return fn(sym, close, volume, panel=panel)
    return wrapped


def lib_frames(panel):
    closes, volumes, grid = panel['closes'], panel['volumes'], panel['grid']
    lib = {
        'mom_10d_skip5': lambda s, c, v: c.shift(5) / c.shift(15) - 1.0,
        'mom_120d_skip5': lambda s, c, v: c.shift(5) / c.shift(125) - 1.0,
        'vix_beta_cond_60x20': bench_beta_cond('VIX', sign=-1.0),
        'vol_of_vol20x60': lambda s, c, v: c.pct_change().rolling(20).std().rolling(60).std(),
    }
    out = {}
    for lbl, fn in lib.items():
        out[lbl] = factor_values(closes, volumes, grid, with_panel(fn, panel))
    return out


def max_lib_corr(fac, libs, cd):
    best = 0.0
    for lbl, lf in libs.items():
        cs = []
        for t in cd:
            x, y = fac.loc[t], lf.loc[t]
            mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= MIN_VALID:
                r = pd.Series(x[mask]).corr(pd.Series(y[mask]), method='spearman')
                if np.isfinite(r):
                    cs.append(r)
        if cs:
            best = max(best, abs(float(np.mean(cs))))
    return best


if __name__ == '__main__':
    panel = build_panel()
    closes, volumes, grid = panel['closes'], panel['volumes'], panel['grid']
    ret = forward_returns(closes, grid, H)
    libs = lib_frames(panel)
    cd = grid

    print(f'=== BATCH-3a-FIX MACRO SPILLOVER h={H} ===')
    for label, fn in CANDIDATES:
        fac = factor_values(closes, volumes, grid, with_panel(fn, panel))
        cov = float(fac.notna().mean().mean())
        f10 = fac.iloc[::10]
        turn = float(f10.rank(axis=1).diff().abs().mean().mean()) if len(f10) > 2 else np.nan
        ics = daily_ic(fac, ret, min_valid=MIN_VALID)
        m = summarize(ics, label, H)
        corr = max_lib_corr(fac, libs, cd)
        print(f'   cov={cov:.3f} turn={turn:.3f} max_lib_corr={corr:.3f}')
        if len(ics) > 0:
            s = ics['ic']
            print('   yearly:', {int(y): round(float(v), 4) for y, v in s.groupby(s.index.year).mean().items()})
