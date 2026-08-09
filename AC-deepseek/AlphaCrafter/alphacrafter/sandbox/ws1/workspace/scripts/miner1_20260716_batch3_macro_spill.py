"""miner_1 batch-3a: cross-asset macro-spillover conditional beta factors.

Idea: asset's rolling beta to a macro/benchmark series, scaled by that
benchmark's own momentum (like the admitted vix_beta_cond_60x20 but for other
macros: USDJPY carry-risk, WTI/COPPER commodity cycle, US10Y/CN10Y rates,
SPX equity beta, XAU). These should be near-orthogonal to the 4-factor library.

Admission gate at h=10: |IC| >= 0.007 and |ICIR| >= 0.084.
Also requires max_abs_library_correlation < 0.5.
"""
import numpy as np
import pandas as pd
from miner1_20260716_lib import (build_panel, factor_values, forward_returns,
                                 daily_ic, summarize)

H = 10
MIN_VALID = 8
LIB_FACTORS = ['mom_10d_skip5', 'mom_120d_skip5', 'vix_beta_cond_60x20', 'vol_of_vol20x60']


def macro_beta_cond(macro_key, sign=1.0, win=60, mom=20, demean=False):
    """factor = sign * rolling_beta(asset_ret, macro_ret, win) * macro_mom(mom)."""
    def fn(sym, close, volume, panel=None):
        if panel is None:
            return None
        macro = panel['macro'].get(macro_key)
        if macro is None:
            return None
        grid = panel['grid']
        r_a = close.pct_change().reindex(grid)
        r_m = macro.pct_change().reindex(grid)
        beta = r_a.rolling(win, min_periods=30).cov(r_m) / r_m.rolling(win, min_periods=30).var()
        mm = (macro.reindex(grid) / macro.shift(mom).reindex(grid) - 1.0)
        f = (sign * beta * mm).replace([np.inf, -np.inf], np.nan)
        if demean:
            f = f - f.rolling(252, min_periods=60).mean()
        return f
    return fn


def spx_beta_cond(sign=1.0, win=60, mom=20):
    """Beta to SPX (equity-market beta) * SPX momentum. SPX is in watchlist."""
    def fn(sym, close, volume, panel=None):
        if panel is None or sym == 'SPX':
            return None
        grid = panel['grid']
        spx = panel['closes']['SPX'].reindex(grid)
        r_a = close.pct_change().reindex(grid)
        r_m = spx.pct_change()
        beta = r_a.rolling(win, min_periods=30).cov(r_m) / r_m.rolling(win, min_periods=30).var()
        mm = (spx / spx.shift(mom) - 1.0)
        return (sign * beta * mm).replace([np.inf, -np.inf], np.nan)
    return fn


CANDIDATES = [
    ('usdjpy_beta_cond_60x20', macro_beta_cond('USDJPY', sign=1.0)),
    ('wti_beta_cond_60x20', macro_beta_cond('WTI', sign=1.0)),
    ('copper_beta_cond_60x20', macro_beta_cond('COPPER', sign=1.0)),
    ('us10y_beta_cond_60x20', macro_beta_cond('US10Y', sign=1.0)),
    ('cn10y_beta_cond_60x20', macro_beta_cond('CN10Y', sign=1.0)),
    ('xau_beta_cond_60x20', macro_beta_cond('XAU', sign=1.0)),
    ('spx_beta_cond_60x20', spx_beta_cond(sign=1.0)),
    ('usdjpy_beta_demean_60x20', macro_beta_cond('USDJPY', sign=1.0, demean=True)),
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
        'vix_beta_cond_60x20': macro_beta_cond('VIX', sign=-1.0),
        'vol_of_vol20x60': lambda s, c, v: c.pct_change().rolling(20).std().rolling(60).std(),
    }
    out = {}
    for lbl, fn in lib.items():
        ffn = with_panel(fn, panel) if lbl == 'vix_beta_cond_60x20' else fn
        out[lbl] = factor_values(closes, volumes, grid, ffn)
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
    cd = grid  # common dates via union grid; factors reindexed already

    print(f'=== BATCH-3a MACRO SPILLOVER h={H} | gate |IC|>=0.007 |ICIR|>=0.084 corr<0.5 ===')
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
