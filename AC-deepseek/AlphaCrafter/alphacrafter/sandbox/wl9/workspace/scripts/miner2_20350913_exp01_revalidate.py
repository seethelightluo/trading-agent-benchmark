"""miner_2 2035-09-13: re-validate all EFFECTIVE lib factors + explore fresh candidates.
Visible window through 2035-08-15. Cross-sectional rank IC vs 10d forward on 15-asset universe.
"""
import pandas as pd, numpy as np, json
from miner2_20350816_toolkit import (load_panel, build_frame, load_macro,
    compute_forward_returns, rank_ic, summarize, rbeta, rcorr)

VISIBLE = "2035-08-15"
uni = load_panel(VISIBLE)
close = build_frame(uni)
ret = close.pct_change()
fwd_10 = compute_forward_returns(close, 10)
print(f"close rows={len(close)} assets={close.shape[1]} range {close.index[0].date()}..{close.index[-1].date()}", flush=True)

M = load_macro(VISIBLE)
d_vix = M['M_VIX']; d_dxy = M['M_DXY']; d_cny = M['M_USDCNY']
d_jpy = M['M_USDJPY']; d_eur = M['M_EURUSD']

def reg_summary(name, fdf, fwd=fwd_10, window=520):
    r = rank_ic(fdf, fwd)
    if len(r['series']) >= 20:
        return summarize(name, r, window)
    else:
        print(f"{name:28s}: TOO FEW ({r['n_ic_dates']})")
        return None

print("\n=== ESTABLISHED LIBRARY (full-window + recent 520d) ===\n", flush=True)
v20 = ret.rolling(20).std(); v120m = v20.rolling(120).mean(); v120s = v20.rolling(120).std(ddof=0)
F = {}
F['mom_10d_skip5'] = close/close.shift(10)-1
F['mom_120d_skip5'] = close/close.shift(120)-1
ma20 = close.rolling(20).mean(); sd20 = close.rolling(20).std()
F['bb_width_20d'] = (2*sd20)/ma20
F['vol_z_20d'] = (v20-v120m)/v120s
F['skew_20d'] = ret.rolling(20).skew()
F['kurt_20d'] = ret.rolling(20).kurt()
def ac(s, w=120):
    return s.rolling(w, min_periods=60).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)
F['ac1_120d'] = pd.DataFrame({s: ac(ret[s], 120) for s in close.columns})
F['kaufman_eff_20d'] = pd.DataFrame({s: (close[s]-close[s].shift(20)).abs()/close[s].diff().abs().rolling(20).sum() for s in close.columns})
F['vix_beta_60'] = rbeta(ret, d_vix.pct_change(), 60)
F['cny_beta_60'] = rbeta(ret, d_cny, 60)

for name in ['mom_10d_skip5','mom_120d_skip5','bb_width_20d','vol_z_20d',
             'skew_20d','kurt_20d','ac1_120d','kaufman_eff_20d',
             'vix_beta_60','cny_beta_60']:
    reg_summary(name, F[name])

print("\n\n=== NEW CANDIDATES (h=10) ===\n", flush=True)
F['mom_ratio_60_120'] = (close/close.shift(60)-1) / (close/close.shift(120)-1).abs()
F['dist_hi_252'] = close/close.rolling(252).max() - 1.0
down = ret.clip(upper=0); up = ret.clip(lower=0)
F['down_ratio_20'] = down.rolling(20).std() / ret.rolling(20).std()
F['updown_asym_20'] = up.rolling(20).std() / down.rolling(20).std().replace(0, np.nan)
F['jpy_beta_60'] = rbeta(ret, d_jpy.pct_change(), 60)
F['eur_beta_60'] = rbeta(ret, d_eur.pct_change(), 60)
hi = pd.DataFrame({a: uni[a]['high'] for a in uni if 'high' in uni[a]}).sort_index()
lo = pd.DataFrame({a: uni[a]['low'] for a in uni if 'low' in uni[a]}).sort_index()
cl_hl = close.reindex(hi.index)
F['hi_lo_pos'] = ((cl_hl - lo) / (hi - lo).replace(0, np.nan)) - 0.5
F['range_norm_20'] = (hi - lo).rolling(20).mean() / cl_hl
F['dxy_beta_60'] = rbeta(ret, d_dxy.pct_change(), 60)
F['cny_beta_rev'] = rbeta(ret, d_cny.pct_change(), 60)

for name in ['mom_ratio_60_120','dist_hi_252','down_ratio_20','updown_asym_20',
             'jpy_beta_60','eur_beta_60','hi_lo_pos','range_norm_20',
             'dxy_beta_60','cny_beta_rev']:
    reg_summary(name, F[name])

print("\nDONE", flush=True)