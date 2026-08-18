"""miner_3 2035-12-28: candidate factor scan across novel families.
Evaluates candidates vs forward returns (h=1,5,10) with library correlation audit.
New families not in the 14-factor library: macro dollar-beta, skew/crash risk,
risk-adjusted momentum, vol-ratio regime, range-position, volume-confirmed reversal.
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'scripts')
from miner3_eval_lib import load_panel, make_library_factors_full, eval_factor, print_eval

# point eval lib at the fresh panel
import miner3_eval_lib
miner3_eval_lib.PANEL = 'scripts/panel_cache_20351227.pkl'
panel = load_panel()
px = panel['close']; op = panel['open']; hi = panel['high']; lo = panel['low']
vol = panel['vol']; ret = panel['ret']; macro = panel['macro']
lib = make_library_factors_full(panel)
print("library factors:", sorted(lib.keys()))
print("panel:", px.shape, px.index.min().date(), "->", px.index.max().date())
print()

logpx = np.log(px)

# ---- candidate 1: dxy_beta_60d (macro dollar-beta, conditional on DXY trend) ----
dxy = macro['DXY'].reindex(px.index).ffill()
dxy_ret = dxy.pct_change()
dxy_beta = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
for i in range(60, len(ret)):
    a = ret.iloc[i-60:i]; b = dxy_ret.iloc[i-60:i]
    m = a.notna() & b.notna()
    if int(m.sum().sum()) < 10:
        continue
    aa = a[m]; bb = b[m]
    cov = (aa * bb).mean() - aa.mean() * bb.mean()
    var = bb.var()
    if var > 0:
        dxy_beta.iloc[i] = cov / var
dxy_trend = dxy_ret.rolling(20).mean()
cand_dxy_beta_cond = dxy_beta * np.sign(dxy_trend).values[:, None]

# ---- candidate 2: skew_20d (return skewness, crash-risk) ----
cand_skew_20d = ret.rolling(20).skew()

# ---- candidate 3: sharpe_mom_60d (risk-adjusted momentum) ----
mom60 = logpx - logpx.shift(60)
vol60 = ret.rolling(60).std()
cand_sharpe_mom_60d = mom60 / vol60

# ---- candidate 4: vol_ratio_5_60 (short/long vol regime) ----
vol5 = ret.rolling(5).std(); vol60b = ret.rolling(60).std()
cand_vol_ratio_5_60 = vol5 / vol60b

# ---- candidate 5: stoch_k_10d (range position, mean-reversion dir -) ----
hi10 = hi.rolling(10).max(); lo10 = lo.rolling(10).min()
cand_stoch_k_10d = -(px - lo10) / (hi10 - lo10)

# ---- candidate 6: vol_surge_rev_2d (reversal confirmed by volume surge) ----
vz = vol / vol.rolling(20).mean()
rev2 = -(logpx - logpx.shift(2))
cand_volsurge_rev_2d = rev2 * (vz > 1.5)

# ---- candidate 7: dxy_mom_60d_cond (asset sensitivity proxy: asset-dxy correlation) ----
cand_dxy_corr_60d = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
for i in range(60, len(ret)):
    a = ret.iloc[i-60:i]; b = dxy_ret.iloc[i-60:i]
    m = a.notna() & b.notna()
    if int(m.sum().sum()) < 10:
        continue
    aa = a[m]; bb = b[m]
    sa, sb = aa.std(), bb.std()
    if sa > 0 and sb > 0:
        cand_dxy_corr_60d.iloc[i] = ((aa * bb).mean() - aa.mean() * bb.mean()) / (sa * sb)

cands = {
    'dxy_beta_cond_60x20': cand_dxy_beta_cond,
    'skew_20d': cand_skew_20d,
    'sharpe_mom_60d': cand_sharpe_mom_60d,
    'vol_ratio_5_60': cand_vol_ratio_5_60,
    'stoch_k_10d': cand_stoch_k_10d,
    'vol_surge_rev_2d': cand_volsurge_rev_2d,
    'dxy_corr_60d': cand_dxy_corr_60d,
}
results = {}
for name, fac in cands.items():
    res = eval_factor(fac, px, horizons=(1, 5, 10), min_valid=8, lib=lib)
    results[name] = res
    print_eval(name, res)
    print()
