"""miner_2 2035-10-11: re-validate ALL EFFECTIVE lib factors + explore fresh candidates.
Visible window through 2035-10-10. Cross-sectional rank IC vs 10d fwd on 15-asset universe.
OOS=2026-07-16 onward. Offline research via persistent csv."""
import pandas as pd, numpy as np, json, os
from miner2_20351011_toolkit import (load_panel, build_frame, load_macro,
    compute_forward_returns, rank_ic, run_summary, rbeta, rcorr)

VISIBLE = "2035-10-10"
OOS_START = "2026-07-16"
H = 10

uni = load_panel(VISIBLE)
close = build_frame(uni)
ret = close.pct_change()
fwd_10 = compute_forward_returns(close, H)
print(f"close rows={len(close)} assets={close.shape[1]} range {close.index[0].date()}..{close.index[-1].date()}", flush=True)

M = load_macro(VISIBLE)
d_vix = M['M_VIX']; d_dxy = M['M_DXY']; d_cny = M['M_USDCNY']; d_jpy = M['M_USDJPY']; d_eur = M['M_EURUSD']

def reg(name, fdf):
    r = rank_ic(fdf, fwd_10)
    if len(r['series']) < 20:
        print(f"{name:30s}: TOO FEW ({r['n_ic_dates']})")
        return
    run_summary(name, r['series'])
    s_oos = r['series'][r['series'].index >= pd.Timestamp(OOS_START)]
    if len(s_oos) >= 15:
        ic = s_oos.mean(); icir = ic/s_oos.std(ddof=1) if s_oos.std(ddof=1)>0 else 0
        print(f"    OOS: ic={ic:+.4f} icir={icir:+.4f} hit={(s_oos>0).mean():.3f} n={len(s_oos)}", flush=True)

v20 = ret.rolling(20).std(); v120m = v20.rolling(120).mean(); v120s = v20.rolling(120).std(ddof=0)

F = {}
F['mom_10d_skip5'] = close/close.shift(10)-1
F['mom_120d_skip5'] = close/close.shift(120)-1
ma20 = close.rolling(20).mean(); sd20 = close.rolling(20).std()
F['bb_width_20d'] = (2*sd20)/ma20
F['vol_z_20d'] = (v20-v120m)/v120s
F['skew_20d'] = ret.rolling(20).skew()
F['kurt_20d'] = ret.rolling(20).kurt()
def ac(s,w=120):
    return s.rolling(w,min_periods=60).apply(lambda x: pd.Series(x).autocorr(lag=1),raw=False)
F['ac1_120d'] = pd.DataFrame({s: ac(ret[s],120) for s in close.columns})
F['kaufman_eff_20d'] = pd.DataFrame({s:(close[s]-close[s].shift(20)).abs()/close[s].diff().abs().rolling(20).sum() for s in close.columns})
F['vix_beta_60'] = rbeta(ret, d_vix.pct_change(), 60)
F['cny_beta_60'] = rbeta(ret, d_cny, 60)
ret_full = ret
dxy_re = d_dxy.reindex(ret.index).fillna(0.0)
corr_sh = pd.DataFrame({s: ret[s].rolling(20).corr(dxy_re[s] if False else dxy_re) for s in close.columns})
# proper per-asset corr with common macro
mac_align = dxy_re
corr_short = pd.DataFrame({s: ret[s].rolling(20).corr(mac_align) for s in close.columns})
corr_long = pd.DataFrame({s: ret[s].rolling(60).corr(mac_align) for s in close.columns})
F['dxy_corr_change_20_60'] = corr_short - corr_long

print("="*70, flush=True)
print("PART 1: ESTABLISHED EFEFCTIVE LIBRARY FACTORS", flush=True)
print("="*70, flush=True)
for name in ['mom_10d_skip5','mom_120d_skip5','bb_width_20d','vol_z_20d','skew_20d',
             'kurt_20d','ac1_120d','kaufman_eff_20d','vix_beta_60','cny_beta_60',
             'dxy_corr_change_20_60']:
    reg(name, F[name])

# Supplementary library factors
days_since_high = {}
for s in close.columns:
    cs = close[s]; rmax = cs.rolling(60,min_periods=1).max()
    out = pd.Series(np.nan, index=cs.index); last = None
    for i in range(len(cs)):
        if cs.iloc[i] >= rmax.iloc[i]: last = cs.index[i]
        if last is not None: out.iloc[i] = (cs.index[i]-last).days
    days_since_high[s] = out
F['days_since_high_60'] = pd.DataFrame(days_since_high)

def streak_len(s,w=14):
    out = pd.Series(np.nan,index=s.index)
    for i in range(w,len(s)):
        win=s.iloc[i-w+1:i+1]; c=0
        for j in range(len(win)-1,-1,-1):
            if win.iloc[j]>0: c+=1
            else: break
        out.iloc[i]=c
    return out
F['streak_len_14'] = pd.DataFrame({s: streak_len(ret[s],14) for s in close.columns})
F['rng_pos_20d'] = (close.close-close.close.shift(20))/close.close.shift(20) if False else (close/close.shift(20)-1)
hi_ = pd.DataFrame({a: uni[a]['high'] for a in uni if 'high' in uni[a]}).sort_index()
lo_ = pd.DataFrame({a: uni[a]['low'] for a in uni if 'low' in uni[a]}).sort_index()
cl_hl = close.reindex(hi_.index)
F['rng_pos_20d'] = ((cl_hl-lo_)/(hi_-lo_).replace(0,np.nan))-0.5

print("="*70, flush=True)
print("PART 1b: OTHER LIBRARY FACTORS", flush=True)
print("="*70, flush=True)
for name in ['days_since_high_60','streak_len_14','rng_pos_20d']:
    if name in F:
        reg(name, F[name])