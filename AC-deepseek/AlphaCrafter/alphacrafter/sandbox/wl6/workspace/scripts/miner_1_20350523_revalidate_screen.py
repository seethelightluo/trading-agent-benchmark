"""miner_1 2035-05-23: revalidate full library + screen new candidates.
VIS auto = 2035-05-22 via date.json. Uses fastlib vectorized rank IC (Pearson on ranks).
Gate: |IC10|>=0.0070 and |ICIR10|>=0.0840.
"""
import sys, os, json
sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd
import importlib.util
spec = importlib.util.spec_from_file_location("fl", "scripts/miner_1_20281102_fastlib.py")
fl = importlib.util.module_from_spec(spec); spec.loader.exec_module(fl)

close = fl.load_close_panel()
ret = close.pct_change()
macro = fl.load_macro_panel()
fwd = fl.forward_returns(close, horizons=(1,2,3,5,10,20))
vixr = macro['VIX'].pct_change()
print(f"Revalidation through {close.index[-1].date()} | rows={len(close)} | macro={macro.shape}", flush=True)

# ---- existing library factor signals ----
def sig_mom10(): return close.shift(5)/close.shift(15)-1.0
def sig_mom120(): return close.shift(5)/close.shift(125)-1.0
def sig_vov(): return ret.rolling(20).std().rolling(60).std()
def sig_lowvol(): return -ret.rolling(20).std()
def sig_betavix_neg():
    b = ret.rolling(60).cov(vixr)/vixr.rolling(60).var(); return -b
def sig_vixcond():
    b = ret.rolling(60).cov(vixr)/vixr.rolling(60).var()
    return -b*(macro['VIX']/macro['VIX'].shift(20)-1.0)
def sig_dvr():
    down = ret.clip(upper=0).rolling(20).std(); return down/ret.rolling(120).std()
def sig_betacn10():
    r10 = close['CN10Y'].pct_change(); return ret.rolling(60).cov(r10)/r10.rolling(60).var()
def sig_betachi():
    rhi = close['HSI'].pct_change(); return ret.rolling(60).cov(rhi)/rhi.rolling(60).var()
def sig_corr10y():
    r10 = close['US10Y'].pct_change(); return ret.rolling(60).corr(r10)
def sig_skew(): return -ret.rolling(20).skew()
def sig_vovchg():
    v = ret.rolling(20).std(); return v.diff(20)/v.rolling(20).mean()
def sig_xaucop():
    cond = ((close['XAU'].pct_change(20)>0)&(close['COPPER'].pct_change(20)>0)).astype(float)
    return cond.mul(-1.0, axis=0)
def sig_volbeta():
    v = ret.rolling(20).std(); spx_v = ret['SPX'].rolling(20).std()
    return v.rolling(60).cov(spx_v)/spx_v.rolling(60).var()
def sig_signewma():
    return (close/close.ewm(span=60).mean()-1.0).apply(np.sign)

library = {
 'mom_10d_skip5': sig_mom10,'mom_120d_skip5': sig_mom120,'vol_of_vol20x60': sig_vov,
 'low_vol_20d': sig_lowvol,'beta_vix_60d_neg': sig_betavix_neg,'vix_beta_cond_60x20': sig_vixcond,
 'down_vol_ratio_20x120': sig_dvr,'beta_cn10y_60d': sig_betacn10,'beta_chi_60d': sig_betachi,
 'corr_us10y_60d': sig_corr10y,'skew_20d_neg': sig_skew,'vol_of_vol_chg_20d': sig_vovchg,
 'xau_copper_cond_20d': sig_xaucop,'vol_beta_spx_60d