"""miner_1 2026-11-19: batch exploration of candidate factor families.
Each candidate computed on the 15-asset cross-section through visible_through.
Admission gates: |IC| >= 0.0070, |ICIR| >= 0.0840 at 10d horizon.
"""
import sys; sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd
from miner_1_common import *

panel, vpanel = load_panel()
ret = panel.pct_change()
fwd10 = forward_returns(panel, 10)

def rolling_apply_skew(s, w):
    return s.rolling(w).skew()

def corr_to_macro(asset_ret, macro_ret, w=60):
    a = asset_ret.copy()
    m = macro_ret.reindex(a.index).ffill()
    return a.rolling(w).corr(m)

cands = {}

# F1: 60d vol-adjusted momentum (60d Sharpe-like)
mu = ret.rolling(60).mean()
sd = ret.rolling(60).std()
cands['vol_adj_mom_60'] = mu / sd

# F2: downside vol share
def downside_share(s, w=60):
    neg = s.where(s < 0, 0.0)
    dd = np.sqrt((neg**2).rolling(w).mean())
    tot = s.rolling(w).std()
    return dd / tot
cands['downside_vol_share_60'] = ret.apply(downside_share)

# F3: skewness 30d
cands['skew_30'] = ret.rolling(30).skew()

# F4-F6: macro-correlation betas
usdjpy = load_macro_panel('USDJPY').pct_change()
eurusd = load_macro_panel('EURUSD').pct_change()
dxy = load_macro_panel('DXY').pct_change()
cands['corr_usdjpy_60'] = ret.apply(lambda s: corr_to_macro(s, usdjpy, 60))
cands['corr_eurusd_60'] = ret.apply(lambda s: corr_to_macro(s, eurusd, 60))
cands['corr_dxy_60'] = ret.apply(lambda s: corr_to_macro(s, dxy, 60))

# F7: z-distance from MA60
ma60 = panel.rolling(60).mean()
std60 = ret.rolling(60).std()
cands['z_dist_ma60'] = (panel / ma60 - 1.0) / std60

# F8: 20d max drawdown
def max_dd_20(s):
    roll_max = s.rolling(20).max()
    return s / roll_max - 1.0
cands['max_dd_20'] = panel.apply(max_dd_20)

# F9: 10d momentum of 20d momentum (momentum acceleration)
cands['mom_accel_10'] = (panel.pct_change(20) - panel.pct_change(20).shift(10))

# F10: 20d high-low range position
ll20 = panel.rolling(20).min()
hh20 = panel.rolling(20).max()
cands['range_pos_20'] = (panel - ll20) / (hh20 - ll20)

# F11: 60d beta to USDJPY
def beta_to(s, m, w=60):
    a = s.copy(); mm = m.reindex(a.index).ffill()
    cov = a.rolling(w).cov(mm)
    var = mm.rolling(w).var()
    return cov / var
cands['beta_usdjpy_60'] = ret.apply(lambda s: beta_to(s, usdjpy, 60))

# F12: 60d beta to EURUSD
cands['beta_eurusd_60'] = ret.apply(lambda s: beta_to(s, eurusd, 60))

print(f"{'factor':<24}{'IC':>8}{'ICIR':>8}{'n':>6}{'hit':>7}{'t':>7}{'cov':>7}")
results = {}
for name, fser in cands.items():
    ics = spearman_ic_series(fser, fwd10)
    m = ic_metrics(ics)
    cov = coverage(fser, panel)
    results[name] = (m, cov)
    print(f"{name:<24}{m['ic']:>8.4f}{m['icir']:>8.4f}{m['n_ic_dates']:>6}{m['hit']:>7.3f}{m['tstat']:>7.2f}{cov:>7.3f}")