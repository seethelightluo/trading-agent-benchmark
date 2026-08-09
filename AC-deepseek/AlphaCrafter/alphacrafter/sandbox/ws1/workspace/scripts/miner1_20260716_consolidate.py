"""miner_1: final consolidation for passing candidates (h=10 admission).
Computes IC/ICIR/hit, coverage, 10d-rebalance rank turnover, and a vectorized
pairwise cross-sectional rank-correlation matrix. Prints max_abs_library_corr
for each factor in the order they will be persisted.
"""
import time
import numpy as np
import pandas as pd
from miner1_20260716_lib import (build_panel, factor_values, forward_returns,
                                 daily_ic)

H = 10
MIN_VALID = 8
t0 = time.time()

panel = build_panel()
closes, volumes, grid = panel['closes'], panel['volumes'], panel['grid']
print(f'panel built in {time.time()-t0:.0f}s  grid_dates={len(grid)}')

CANDIDATES = {
    'mom_10d_skip5': lambda s, c, v: c.shift(5) / c.shift(15) - 1.0,
    'mom_20d_skip5': lambda s, c, v: c.shift(5) / c.shift(25) - 1.0,
    'mom_120d_skip5': lambda s, c, v: c.shift(5) / c.shift(125) - 1.0,
    'trend_sma60': lambda s, c, v: c / c.rolling(60).mean() - 1.0,
    'trend_sma120': lambda s, c, v: c / c.rolling(120).mean() - 1.0,
    'risk_adj_trend20': lambda s, c, v: (c.pct_change().rolling(20).mean()
                                         / c.pct_change().rolling(20).std()
                                         ).replace([np.inf, -np.inf], np.nan),
    'vol_of_vol20x60': lambda s, c, v: c.pct_change().rolling(20).std().rolling(60).std(),
    'zscore_rev_20d': lambda s, c, v: ((c - c.rolling(20).mean())
                                       / (c.pct_change().rolling(20).std() * c)
                                       ).replace([np.inf, -np.inf], np.nan),
    'inv_vol_60d': lambda s, c, v: (-c.pct_change().rolling(60).std()
                                    ).replace([np.inf, -np.inf], np.nan),
    'vix_beta_cond_60x20': None,  # built below with macro
}


def vix_beta_cond(sym, close, volume, panel):
    macro = panel['macro'].get('VIX')
    if macro is None:
        return None
    grid = panel['grid']
    r_a = close.pct_change().reindex(grid)
    r_m = macro.pct_change().reindex(grid)
    beta = r_a.rolling(60, min_periods=30).cov(r_m) / r_m.rolling(60, min_periods=30).var()
    mm = (macro.reindex(grid) / macro.shift(20).reindex(grid) - 1.0)
    return (-1.0 * beta * mm).replace([np.inf, -np.inf], np.nan)


frames = {}
for label, fn in CANDIDATES.items():
    if fn is None:
        frames[label] = factor_values(closes, volumes, grid,
                                      lambda s, c, v, p=panel: vix_beta_cond(s, c, v, p))
    else:
        frames[label] = factor_values(closes, volumes, grid, fn)

ret = forward_returns(closes, grid, H)
print('\n=== ADMISSION METRICS h=10 ===')
metrics = {}
for label, fac in frames.items():
    ics = daily_ic(fac, ret, min_valid=MIN_VALID)
    ic = ics['ic']
    mean_ic = float(ic.mean())
    std_ic = float(ic.std(ddof=1))
    icir = mean_ic / std_ic if std_ic > 0 else np.nan
    hit = float((ic > 0).mean())
    cov_assets = float(fac.notna().mean().mean())
    cov_dates8 = float((fac.notna().sum(axis=1) >= MIN_VALID).mean())
    f10 = fac.iloc[::10].rank(axis=1)
    turn = float(f10.diff().abs().mean().mean()) if len(f10) > 2 else np.nan
    metrics[label] = dict(ic=mean_ic, icir=icir, hit=hit, dates=len(ic),
                          cov_assets=cov_assets, cov_dates8=cov_dates8, turn=turn)
    print(f'[{label}] dates={len(ic)} IC={mean_ic:+.4f} ICIR={icir:+.3f} hit={hit:.2f} '
          f'cov_asset={cov_assets:.3f} cov_dates8+={cov_dates8:.3f} turn={turn:.3f}')

print('\n=== PAIRWISE RANK CORR (vectorized, min 8 valid dates) ===')
ranks = {l: f.rank(axis=1) for l, f in frames.items()}
labels = list(CANDIDATES.keys())
corr = pd.DataFrame(index=labels, columns=labels, dtype=float)
for a in labels:
    for b in labels:
        if a == b:
            corr.loc[a, b] = 1.0
            continue
        dfa = ranks[a].where(ranks[a].notna() & ranks[b].notna())
        dfb = ranks[b].where(ranks[a].notna() & ranks[b].notna())
        nvalid = dfa.notna().sum(axis=1)
        ok = nvalid >= MIN_VALID
        if ok.sum() == 0:
            corr.loc[a, b] = np.nan
            continue
        v = dfa[ok].corrwith(dfb[ok], axis=1).mean()
        corr.loc[a, b] = v
pd.set_option('display.width', 250)
print(corr.round(3))

print('\n=== max_abs_library_correlation (persistence order) ===')
persist_order = labels  # persistence order = this list
persisted = []
for l in persist_order:
    if not persisted:
        print(f'{l}: 0.0 (first in library)')
    else:
        mx = max(abs(corr.loc[l, p]) for p in persisted)
        print(f'{l}: max_abs_lib_corr = {mx:.3f} (vs {persisted})')
    persisted.append(l)

print(f'\ntotal runtime {time.time()-t0:.0f}s')
