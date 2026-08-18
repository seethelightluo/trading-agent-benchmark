"""miner2: per-year IC breakdown for existing factors + dxy fix + regime split.
Determines when sign flips occurred and whether drift is regime-driven.
"""
import sys, json
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from miner2_20291009_lib import load_prices, load_macro, cross_sectional_ic, forward_ret, WATCH

prices, closes = load_prices(3000)
macro = load_macro(3000)
sim_end = pd.Timestamp('2029-10-08')
macro = macro[macro.index <= sim_end]

def rel_mom(closes, w=20, s=5):
    mom = closes / closes.shift(s) - 1.0
    return mom.rolling(w).mean().sub(mom.rolling(w).mean().median(axis=1), axis=0)

def dvol(closes, w=20):
    ret = closes.pct_change()
    dret = ret.where(ret < 0, 0.0)
    return dret.rolling(w).std() / ret.rolling(w).std().replace(0, np.nan)

def beta_ew(closes, w=60):
    ret = closes.pct_change(); mkt = ret.mean(axis=1)
    return pd.DataFrame({c: ret[c].rolling(w).cov(mkt) / mkt.rolling(w).var().replace(0, np.nan) for c in closes.columns}, index=closes.index)

def maxret(closes, w=20):
    return closes.pct_change().rolling(w).max()

def corr_ew(closes, w=60):
    ret = closes.pct_change(); mkt = ret.mean(axis=1)
    return pd.DataFrame({c: ret[c].rolling(w).corr(mkt) for c in closes.columns}, index=closes.index)

def kurt(closes, w=20, s=5):
    return closes.pct_change().rolling(w).kurt().shift(s)

def dxy_beta_cond(closes, macro, w=60, m=20):
    ret = closes.pct_change()
    dxy = macro['DXY'].reindex(closes.index).ffill()
    dxyr = dxy.pct_change(); dmom = dxy.pct_change(m)
    return pd.DataFrame({c: (ret[c].rolling(w).cov(dxyr) / dxyr.rolling(w).var().replace(0, np.nan)) * dmom for c in closes.columns}, index=closes.index)

factors = {
    'rel_mom_20d_skip5': (rel_mom, {'w':20,'s':5}, 1),
    'downside_vol_ratio_20': (dvol, {'w':20}, 1),
    'beta_ew_60d': (beta_ew, {'w':60}, 1),
    'max_ret_20d': (maxret, {'w':20}, 1),
    'corr_ew_60': (corr_ew, {'w':60}, 1),
    'kurt_20d_skip5': (kurt, {'w':20,'s':5}, 1),
    'dxy_beta_cond_60x20': (dxy_beta_cond, {'w':60,'m':20}, -1),
}
fwd10 = forward_ret(closes, 10)
out = {}
for fid, (fn, params, direction) in factors.items():
    fdf = fn(closes, **params) if fid != 'dxy_beta_cond_60x20' else fn(closes, macro, **params)
    ic = cross_sectional_ic(fdf, fwd10)
    print(f'--- {fid} (admitted dir {direction}) ---')
    yearly = {}
    for yr in sorted(set(ic.index.year)):
        sub = ic[ic.index.year == yr] * direction
        if len(sub):
            yearly[yr] = (round(float(sub.mean()), 4), round(float(sub.mean()/sub.std(ddof=1)), 3) if sub.std(ddof=1) > 0 else np.nan, int(len(sub)))
            print(f'  {yr}: IC={yearly[yr][0]:+.4f} ICIR={yearly[yr][1]:+.3f} n={yearly[yr][2]}')
    # regime split: VIX > 30 (risk-off) vs <= 30
    vix = macro['VIX'].reindex(ic.index).ffill()
    off = ic[vix > 30] * direction
    on = ic[vix <= 30] * direction
    print(f'  RISK-OFF (VIX>30): IC={off.mean():+.4f} ICIR={off.mean()/off.std(ddof=1):+.3f} n={len(off)}' if len(off) > 5 else f'  RISK-OFF n={len(off)}')
    print(f'  RISK-ON  (VIX<=30): IC={on.mean():+.4f} ICIR={on.mean()/on.std(ddof=1):+.3f} n={len(on)}' if len(on) > 5 else f'  RISK-ON n={len(on)}')
    out[fid] = {'yearly': {str(k): v for k, v in yearly.items()}}
    print()

with open('scripts/miner2_20291009_regime_drift.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
