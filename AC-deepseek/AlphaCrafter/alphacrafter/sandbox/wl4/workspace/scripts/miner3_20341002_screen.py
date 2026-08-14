"""miner_3 exploratory screen (2034-10-02 cycle): quick IC/ICIR of candidate factor ideas.
Screening only - formal validation of promising candidates happens in dedicated scripts.
Data through 2034-09-29 only.
"""
import sys, os
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from miner3_20341002_common import load_assets, load_macro, rank_ic_panel, build_forward_returns

px, rt = load_assets()
macro = load_macro()
print(f'assets: {px.shape[1]}, price rows {px.shape[0]} ({px.index.min().date()}..{px.index.max().date()})')
print(f'macro cols: {list(macro.columns)} rows {len(macro)}')

fwd = build_forward_returns(px, horizons=(10,))

def screen(name, fdf):
    ic = rank_ic_panel(fdf, fwd[10], min_valid=8)
    if len(ic) < 30:
        print(f'{name:28s} n_dates={len(ic):5d}  INSUFFICIENT')
        return
    icm, ics = ic.mean(), ic.std()
    print(f'{name:28s} n={len(ic):5d} ic={icm:+.4f} icir={icm/ics:+.3f} hit={float((ic>0).mean()):.2f} '
          f'cov={float(fdf.notna().sum().sum()/fdf.size):.2f} r250_ic={float(ic[ic.index>ic.index[-1]-pd.Timedelta(days=400)].mean()):+.4f}')

# --- candidate 1: Kaufman efficiency ratio 20d ---
er20 = (px - px.shift(20)).abs() / rt.abs().rolling(20).sum()
screen('efficiency_ratio_20d', er20)
# --- candidate 2: efficiency ratio 40d ---
er40 = (px - px.shift(40)).abs() / rt.abs().rolling(40).sum()
screen('efficiency_ratio_40d', er40)
# --- candidate 3: DXY beta 60d ---
dxy_ret = macro['DXY'].pct_change()
aligned_rt = rt.reindex(dxy_ret.index)
cov = aligned_rt.rolling(60).cov(dxy_ret)
var = dxy_ret.rolling(60).var()
dxy_beta = cov / var
screen('dxy_beta_60d', dxy_beta)
# --- candidate 4: USDJPY beta 60d ---
jpy_ret = macro['USDJPY'].pct_change()
cov2 = rt.reindex(jpy_ret.index).rolling(60).cov(jpy_ret)
var2 = jpy_ret.rolling(60).var()
jpy_beta = cov2 / var2
screen('usdjpy_beta_60d', jpy_beta)
# --- candidate 5: rolling sharpe 60d ---
sh60 = rt.rolling(60).mean() / rt.rolling(60).std()
screen('sharpe_60d', sh60)
# --- candidate 6: up/down capture 60d ---
mkt = rt.mean(axis=1)
up = (mkt > 0).astype(float)
dn = (mkt < 0).astype(float)
up_avg = (rt * up).rolling(60).sum() / up.rolling(60).sum()
dn_avg = (rt * dn).rolling(60).sum() / dn.rolling(60).sum()
cap = up_avg / dn_avg.abs()
screen('updown_capture_60d', cap)
# --- candidate 7: yield spread beta 60d (US10Y-CN10Y) ---
spread = px['US10Y'] - px['CN10Y']
sp_ret = spread.pct_change()
cov3 = rt.reindex(sp_ret.index).rolling(60).cov(sp_ret)
var3 = sp_ret.rolling(60).var()
sp_beta = cov3 / var3
screen('yield_spread_beta_60d', sp_beta)
# --- candidate 8: 20d return vs 60d return (momentum slope, different from accel which was vol-adj) ---
slope = (px / px.shift(20) - 1) - (px / px.shift(60) - 1)
screen('mom_slope_20x60_raw', slope)
# --- candidate 9: 10d vol ratio (realized vol contraction) ---
vr = rt.rolling(10).std() / rt.rolling(60).std()
screen('vol_ratio_10_60', vr)
# --- candidate 10: skewness 60d ---
sk = rt.rolling(60).skew()
screen('skew_60d', sk)
