"""miner2 re-validation of existing effective factors on data thru 2029-10-08.
Checks drift vs 2026 warm-up validation. Also prints regime (VIX) context.
"""
import sys, json
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from miner2_20291009_lib import load_prices, load_macro, run_validation, WATCH

prices, closes = load_prices(3000)
macro = load_macro(3000)
print('closes shape:', closes.shape, closes.index[0], closes.index[-1])
print('macro last rows:')
print(macro.tail(3))
vix = macro['VIX']
print('VIX now:', float(vix.iloc[-1]), '| VIX 20d ago:', float(vix.iloc[-21]) if len(vix) > 21 else None)

# ---------------- existing factor definitions (recompute) ----------------
def rel_mom_20d_skip5(closes, w=20, s=5):
    mom = closes / closes.shift(s) - 1.0
    momw = mom.rolling(w).mean()
    return momw.sub(momw.median(axis=1), axis=0)

def downside_vol_ratio_20(closes, w=20):
    ret = closes.pct_change()
    vol = ret.rolling(w).std()
    dret = ret.where(ret < 0, 0.0)
    dvol = dret.rolling(w).std()
    return dvol / vol.replace(0, np.nan)

def beta_ew_60d(closes, w=60):
    ret = closes.pct_change()
    mkt = ret.mean(axis=1)
    out = {}
    for c in closes.columns:
        cov = ret[c].rolling(w).cov(mkt)
        var = mkt.rolling(w).var()
        out[c] = cov / var.replace(0, np.nan)
    return pd.DataFrame(out, index=closes.index)

def max_ret_20d(closes, w=20):
    ret = closes.pct_change()
    return ret.rolling(w).max()

def corr_ew_60(closes, w=60):
    ret = closes.pct_change()
    mkt = ret.mean(axis=1)
    out = {}
    for c in closes.columns:
        out[c] = ret[c].rolling(w).corr(mkt)
    return pd.DataFrame(out, index=closes.index)

def kurt_20d_skip5(closes, w=20, s=5):
    ret = closes.pct_change()
    return ret.rolling(w).kurt().shift(s)

def dxy_beta_cond_60x20(closes, macro, w=60, m=20):
    ret = closes.pct_change()
    dxy = macro['DXY'].reindex(closes.index).ffill().pct_change()
    dmom = macro['DXY'].reindex(closes.index).ffill().pct_change(m)
    out = {}
    for c in closes.columns:
        cov = ret[c].rolling(w).cov(dxy)
        var = dxy.rolling(w).var()
        beta = cov / var.replace(0, np.nan)
        out[c] = beta * dmom
    return pd.DataFrame(out, index=closes.index)

existing = {
    'rel_mom_20d_skip5': (rel_mom_20d_skip5, {'w': 20, 's': 5}, 1),
    'downside_vol_ratio_20': (downside_vol_ratio_20, {'w': 20}, 1),
    'beta_ew_60d': (beta_ew_60d, {'w': 60}, 1),
    'max_ret_20d': (max_ret_20d, {'w': 20}, 1),
    'corr_ew_60': (corr_ew_60, {'w': 60}, 1),
    'kurt_20d_skip5': (kurt_20d_skip5, {'w': 20, 's': 5}, 1),
    'dxy_beta_cond_60x20': (dxy_beta_cond_60x20, {'w': 60, 'm': 20}, -1),
}
results = {}
for fid, (fn, params, direction) in existing.items():
    fdf = fn(closes, **params)
    res = run_validation(fid, fid, fdf, closes, direction=direction, horizon=10,
                         params=params, tags=['revalidation'])
    results[fid] = res
    print()

with open('scripts/miner2_20291009_reval_results.json', 'w') as f:
    json.dump(results, f, indent=1, default=str)
print('saved revalidation results')
