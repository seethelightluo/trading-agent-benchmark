"""Screener cycle 17: full 12-factor library -> regime check, pairwise corr, quality_ic_tilt ensemble.

Library now contains 12 admitted factors (4 original + 8 new from miner batch4/batch5).
Rebuild factor_ensemble.json (max 10 factors) using quality_ic_tilt: q=|IC|*|ICIR|, dir=sign(IC).
"""
import json, os, glob
import numpy as np
import pandas as pd
from miner1_20260716_lib import (build_panel, factor_values, forward_returns,
                                 daily_ic, WATCH)

H = 10
MIN_VALID = 8

panel = build_panel()
closes, volumes, grid = panel['closes'], panel['volumes'], panel['grid']
print(f'panel: grid_dates={len(grid)} assets={len(closes)}')


# ---------------- factor definitions (mirror the persisted library) ----------------
def macro_beta_cond(macro_key, sign=1.0, win=60, mom=20):
    def fn(sym, close, volume, panel=None):
        macro = panel['macro'].get(macro_key)
        if macro is None:
            macro = panel['closes'].get(macro_key)
        if macro is None:
            return None
        g = panel['grid']
        r_a = close.pct_change().reindex(g)
        r_m = macro.pct_change().reindex(g)
        beta = r_a.rolling(win, min_periods=30).cov(r_m) / r_m.rolling(win, min_periods=30).var()
        mm = macro.reindex(g) / macro.shift(mom).reindex(g) - 1.0
        return (sign * beta * mm).replace([np.inf, -np.inf], np.nan)
    return fn


def vix_beta_frame(sym, close, volume):
    macro = panel['macro'].get('VIX')
    g = panel['grid']
    r_a = close.pct_change().reindex(g)
    r_m = macro.pct_change().reindex(g)
    beta = r_a.rolling(60, min_periods=30).cov(r_m) / r_m.rolling(60, min_periods=30).var()
    mm = macro.reindex(g) / macro.shift(20).reindex(g) - 1.0
    return (-1.0 * beta * mm).replace([np.inf, -np.inf], np.nan)


def max_ratio(win=20):
    def fn(sym, close, volume):
        r = close.pct_change().rolling(win)
        return (r.max() / r.min().abs()).replace([np.inf, -np.inf], np.nan)
    return fn


def consec_up_ratio(win=20):
    def fn(sym, close, volume):
        r = (close.pct_change() > 0).astype(float)
        def run_len(x):
            x = x.values
            best_up = best_dn = cur_u = cur_d = 0
            for v in x:
                if v == 1:
                    cur_u += 1; cur_d = 0
                    best_up = max(best_up, cur_u)
                else:
                    cur_d += 1; cur_u = 0
                    best_dn = max(best_dn, cur_d)
            s = best_up + best_dn
            return best_up / s if s > 0 else np.nan
        return r.rolling(win).apply(run_len, raw=False)
    return fn


def autocorr(px, n=20):
    return px.pct_change().rolling(n).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) >= 3 else np.nan, raw=True)


def amihud(win=20):
    def fn(sym, close, volume):
        if volume is None or volume.abs().sum() == 0:
            return None
        illiq = (close.pct_change().abs() / volume.replace(0, np.nan)).rolling(win).mean()
        return (-illiq).replace([np.inf, -np.inf], np.nan)
    return fn


FACTORS = {
    'mom_10d_skip5':      lambda s, c, v: c.shift(5) / c.shift(15) - 1.0,
    'mom_120d_skip5':     lambda s, c, v: c.shift(5) / c.shift(125) - 1.0,
    'vol_of_vol20x60':    lambda s, c, v: c.pct_change().rolling(20).std().rolling(60).std(),
    'vix_beta_cond_60x20': vix_beta_frame,
    'amihud_liquidity_20d': amihud(20),
    'btc_spill_cond_60x20': (lambda f: (lambda s, c, v: f(s, c, v, panel=panel)))(macro_beta_cond('BTC', sign=1.0, win=60, mom=20)),
    'consec_up_ratio_20': consec_up_ratio(20),
    'dxy_cond_60x20':     (lambda f: (lambda s, c, v: f(s, c, v, panel=panel)))(macro_beta_cond('DXY', sign=1.0, win=60, mom=20)),
    'eff_ratio_60d':      lambda s, c, v: (c - c.shift(60)).abs() / c.diff().abs().rolling(60).sum(),
    'max_ratio_20':       max_ratio(20),
    'ret_autocorr_20d':   lambda s, c, v: -autocorr(c, 20),
    'usdjpy_beta_cond_60x20': (lambda f: (lambda s, c, v: f(s, c, v, panel=panel)))(macro_beta_cond('USDJPY', sign=1.0, win=60, mom=20)),
}

frames = {}
for label, fn in FACTORS.items():
    frames[label] = factor_values(closes, volumes, grid, fn)
    cov = frames[label].notna().mean().mean()
    print(f'[{label}] cov_asset_days={cov:.3f}')

# ---------------- validation: full-sample + recent 250d IC ----------------
ret = forward_returns(closes, grid, H)
ics = {}
for label, fac in frames.items():
    a = daily_ic(fac, ret, min_valid=MIN_VALID)
    ics[label] = a
    ic = a['ic'].mean(); icir = ic / a['ic'].std(ddof=1) if a['ic'].std(ddof=1) > 0 else np.nan
    a_recent = a.tail(250)
    ic_r = a_recent['ic'].mean(); icir_r = ic_r / a_recent['ic'].std(ddof=1) if a_recent['ic'].std(ddof=1) > 0 else np.nan
    print(f'[{label}] full IC={ic:+.4f} ICIR={icir:+.3f} n={len(a)} | recent250 IC={ic_r:+.4f} ICIR={icir_r:+.3f}')

# ---------------- pairwise cross-sectional rank correlation (mean over dates) ----------------
labels = list(FACTORS.keys())
corr = pd.DataFrame(index=labels, columns=labels, dtype=float)
common_dates = None
for l in labels:
    d = frames[l].dropna(how='all').index
    common_dates = d if common_dates is None else common_dates.intersection(d)
for i, a in enumerate(labels):
    for j, b in enumerate(labels):
        if i > j:
            continue
        cs = []
        for t in common_dates:
            fa, fb = frames[a].loc[t], frames[b].loc[t]
            mask = fa.notna() & fb.notna() & np.isfinite(fa) & np.isfinite(fb)
            if mask.sum() >= MIN_VALID:
                r = pd.Series(fa[mask]).corr(pd.Series(fb[mask]), method='spearman')
                if np.isfinite(r):
                    cs.append(r)
        v = float(np.mean(cs)) if cs else np.nan
        corr.loc[a, b] = v
        corr.loc[b, a] = v
pd.set_option('display.width', 250)
print('\n=== PAIRWISE CROSS-SECTIONAL RANK CORR (mean over dates) ===')
print(corr.round(2))
corr.to_csv('scripts/screener_cycle17_corr_matrix.csv')
print(f'common dates used: {len(common_dates)}')

# max abs corr per factor
maxc = corr.abs().max(axis=1)
print('\n=== MAX ABS PAIRWISE CORR PER FACTOR ===')
for l in labels:
    print(f'  {l:<26} {maxc[l]:.3f}')
