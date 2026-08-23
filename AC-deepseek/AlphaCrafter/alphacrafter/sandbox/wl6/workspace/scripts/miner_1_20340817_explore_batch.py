"""miner_1 2034-08-17: batch exploration of fresh candidate factors."""
import numpy as np, pandas as pd, importlib.util

spec = importlib.util.spec_from_file_location("fl", "scripts/miner_1_20281102_fastlib.py")
fl = importlib.util.module_from_spec(spec); spec.loader.exec_module(fl)

close = fl.load_close_panel()
ret = close.pct_change()
macro = fl.load_macro_panel()
fwd = fl.forward_returns(close, horizons=(1,2,3,5,10,20))
print(f"explore through {close.index[-1].date()} | rows={len(close)} | macro={macro.shape}", flush=True)

lib = {
 'mom_10d_skip5': lambda: close.shift(5)/close.shift(15)-1.0,
 'mom_120d_skip5': lambda: close.shift(5)/close.shift(125)-1.0,
 'vol_of_vol20x60': lambda: ret.rolling(20).std().rolling(60).std(),
 'low_vol_20d': lambda: -ret.rolling(20).std(),
 'beta_vix_60d_neg': lambda: -ret.rolling(60).cov(macro['VIX'].pct_change())/macro['VIX'].pct_change().rolling(60).var(),
 'vix_beta_cond_60x20': lambda: -(ret.rolling(60).cov(macro['VIX'].pct_change())/macro['VIX'].pct_change().rolling(60).var())*(macro['VIX']/macro['VIX'].shift(20)-1.0),
 'down_vol_ratio_20x120': lambda: ret.clip(upper=0).rolling(20).std()/ret.rolling(120).std(),
 'beta_cn10y_60d': lambda: ret.rolling(60).cov(close['CN10Y'].pct_change())/close['CN10Y'].pct_change().rolling(60).var(),
 'beta_chi_60d': lambda: ret.rolling(60).cov(close['HSI'].pct_change())/close['HSI'].pct_change().rolling(60).var(),
 'corr_us10y_60d': lambda: ret.rolling(60).corr(close['US10Y'].pct_change()),
 'skew_20d_neg': lambda: -ret.rolling(20).skew(),
 'vol_of_vol_chg_20d': lambda: (lambda v: v.diff(20)/v.rolling(20).mean())(ret.rolling(20).std()),
 'xau_copper_cond_20d': lambda: ((close['XAU'].pct_change(20)>0)&(close['COPPER'].pct_change(20)>0)).astype(float).mul(-1.0,axis=0),
 'vol_beta_spx_60d': lambda: (lambda v,sv: v.rolling(60).cov(sv)/sv.rolling(60).var())(ret.rolling(20).std(), ret['SPX'].rolling(20).std()),
 'sign_ewma_60d': lambda: (close/close.ewm(span=60).mean()-1.0).apply(np.sign),
}
lib_signals = {k: v().reindex(close.index) for k,v in lib.items()}

dxy = macro['DXY']; dxy_r = dxy.pct_change()
usdjpy = macro['USDJPY'].pct_change()
spx_r = ret['SPX']; xau_r = ret['XAU']; cop_r = ret['COPPER']

def beta_y(x, y):
    return x.rolling(60).cov(y)/y.rolling(60).var()

candidates = {
 'dxy_beta_60d': beta_y(ret, dxy_r),
 'two_beta_gold_60d': beta_y(ret, spx_r) - beta_y(ret, xau_r),
 'range_amp_20d': -((close.rolling(20).max()-close.rolling(20).min())/close).mean(axis=0).mul(0.0) if False else -((close.rolling(20).max()-close.rolling(20).min())/close),
 'nhigh_ratio_5d': (close >= close.rolling(5).max()).rolling(10).mean(),
 'copper_gold_div_60d': ret.rolling(60).corr(cop_r) - ret.rolling(60).corr(xau_r),
 'vol_rev_5x60': -(ret.rolling(5).std()/ret.rolling(60).std()),
 'usdjpy_beta_60d': beta_y(ret, usdjpy),
 'yld_rel_ret_60d': close.shift(5)/close.shift(65)-1.0 - (close['US10Y'].shift(5)/close['US10Y'].shift(65)-1.0),
}

print("\n=== CANDIDATE SCREENING (primary horizon h=5) ===", flush=True)
radio = {}
for name, f in candidates.items():
    f = f.reindex(close.index)
    ic, n = fl.rank_ic_series(f, fwd[5]); ic = ic.dropna()
    if len(ic) == 0:
        print(f"{name:22s} NO_DATA", flush=True); continue
    icv = float(ic.mean()); icir = float(ic.mean()/ic.std(ddof=1)) if ic.std(ddof=1)>0 else 0.0
    hit = float((ic>0).mean())
    mx, _ = fl.max_abs_library_correlation(f, lib_signals, label='explore')
    gate = abs(icv)>=fl.IC_TH and abs(icir)>=fl.ICIR_TH
    cov = float(f.notna().mean())
    radio[name] = {'ic':icv,'icir':icir,'hit':hit,'mxr':mx,'cov':cov,'gate':gate}
    print(f"{name:22s} IC={icv:+.4f} ICIR={icir:+.3f} hit={hit:.2f} n={len(ic)} "
          f"cov={cov:.2f} max_lib_corr={mx if mx is None else round(mx,3)} gate={'PASS' if gate else 'fail'}", flush=True)

print("\n=== DECAY for passing candidates ===", flush=True)
for name, f in candidates.items():
    if name not in radio or not radio[name]['gate']: continue
    f = f.reindex(close.index)
    dec = {}
    for h in sorted(fwd.keys()):
        ic_h,_ = fl.rank_ic_series(f, fwd[h]); ic_h=ic_h.dropna()
        dec[str(h)] = float(ic_h.mean()) if len(ic_h) else np.nan
    print(f"{name:22s} decay_ic={ {k:round(v,4) for k,v in dec.items()} }", flush=True)