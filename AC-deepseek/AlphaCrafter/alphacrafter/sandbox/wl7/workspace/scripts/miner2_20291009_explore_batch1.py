"""miner2 batch exploration of new factor candidates (2029-10-09).
Tests: vix_beta_cond, dd_60d drawdown reversal, vol_adj reversal, regime-switch
momentum, xau_beta, downside beta. Prints IC/ICIR for both directions.
"""
import sys, json
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from miner2_20291009_lib import (load_prices, load_macro, cross_sectional_ic,
                                 forward_ret, ic_stats, coverage_stats,
                                 turnover_rank, decay_profile, library_corr, WATCH)

prices, closes = load_prices(3000)
macro = load_macro(3000)
sim_end = pd.Timestamp('2029-10-08')
macro = macro[macro.index <= sim_end]
ret = closes.pct_change()
mkt = ret.mean(axis=1)
fwd10 = forward_ret(closes, 10)

def vix_beta_cond(closes, macro, w=60, m=20):
    vix = macro['VIX'].reindex(closes.index).ffill()
    vr = vix.pct_change(); vmom = vix.pct_change(m)
    return pd.DataFrame({c: (ret[c].rolling(w).cov(vr) / vr.rolling(w).var().replace(0, np.nan)) * vmom
                         for c in closes.columns}, index=closes.index)

def dd_60d(closes, w=60):
    return closes / closes.rolling(w).max() - 1.0

def vol_adj_rev(closes, w=10, v=20):
    r = closes.pct_change(w)
    vol = closes.pct_change().rolling(v).std()
    return -r / vol.replace(0, np.nan)

def mom_regime(closes, macro, w=20, s=5, thr=30):
    mom = (closes / closes.shift(s) - 1.0).rolling(w).mean()
    mom = mom.sub(mom.median(axis=1), axis=0)
    vix = macro['VIX'].reindex(closes.index).ffill()
    regime = np.where(vix > thr, -1.0, 1.0)
    return mom.mul(regime, axis=0)

def xau_beta(closes, w=60):
    xr = ret['XAU']
    return pd.DataFrame({c: ret[c].rolling(w).cov(xr) / xr.rolling(w).var().replace(0, np.nan)
                         for c in closes.columns}, index=closes.index)

def down_beta(closes, w=60):
    dret = ret.where(mkt < 0, np.nan)
    dvar = dret.rolling(w).var()
    covs = pd.DataFrame({c: ret[c].where(mkt < 0, np.nan).rolling(w).cov(dret.mean(axis=1)) for c in closes.columns}, index=closes.index)
    return covs / dvar.replace(0, np.nan)

def down_vol_ratio(closes, w=20, v=60):
    dret = ret.where(ret < 0, 0.0)
    return (dret.rolling(w).std() / ret.rolling(w).std().replace(0, np.nan)).rolling(v).mean()

candidates = {
    'vix_beta_cond_60x20': (vix_beta_cond, {'w': 60, 'm': 20}),
    'dd_60d': (dd_60d, {'w': 60}),
    'vol_adj_rev_10x20': (vol_adj_rev, {'w': 10, 'v': 20}),
    'mom_regime_20d': (mom_regime, {'w': 20, 's': 5, 'thr': 30}),
    'xau_beta_60d': (xau_beta, {'w': 60}),
    'down_beta_60d': (down_beta, {'w': 60}),
    'down_vol_ratio_20x60': (down_vol_ratio, {'w': 20, 'v': 60}),
}
results = {}
for fid, (fn, params) in candidates.items():
    needs_macro = fid in ('vix_beta_cond_60x20', 'mom_regime_20d')
    fdf = fn(closes, macro, **params) if needs_macro else fn(closes, **params)
    ic = cross_sectional_ic(fdf, fwd10)
    cov = coverage_stats(fdf)
    turn = turnover_rank(fdf)
    for direction in (1, -1):
        st = ic_stats(ic, direction)
        print(f'{fid:24s} dir={direction:+d} IC={st["ic"]:+.4f} ICIR={st["icir"]:+.4f} '
              f'hit={st["hit"]:.3f} n={st["n"]} cov8={cov["coverage_dates_ge8"]:.2f} turn={turn:.3f}')
    decay = decay_profile(fdf, closes)
    maxabs, pair = library_corr(fdf)
    print(f'    decay={decay} max_lib_corr={maxabs}')
    results[fid] = {'params': params}
    print()

with open('scripts/miner2_20291009_candidates_batch1.json', 'w') as f:
    json.dump(results, f, indent=1, default=str)
