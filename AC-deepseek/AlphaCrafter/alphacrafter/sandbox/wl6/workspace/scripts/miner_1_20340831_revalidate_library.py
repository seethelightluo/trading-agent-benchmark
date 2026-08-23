"""miner_1 2034-08-31: re-validate the full effective factor library through visible_through (2034-08-30)."""
import json, math, importlib.util
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("vlib", "scripts/miner_1_20260730_validation_lib.py")
vlib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vlib)

close = vlib.load_close_panel()
ret = close.pct_change()
macro = vlib.load_macro_panel()
fwd = vlib.forward_returns(close, horizons=(1, 2, 3, 5, 10, 20))

def sig_mom10(): return close.shift(5) / close.shift(15) - 1.0
def sig_mom120(): return close.shift(5) / close.shift(125) - 1.0
def sig_vov(): return ret.rolling(20).std().rolling(60).std()
def sig_lowvol(): return -ret.rolling(20).std()
def sig_betavix_neg():
    vixr = macro['VIX'].pct_change()
    beta = ret.rolling(60).cov(vixr) / vixr.rolling(60).var()
    return -beta
def sig_vixcond():
    vixr = macro['VIX'].pct_change()
    beta = ret.rolling(60).cov(vixr) / vixr.rolling(60).var()
    return -beta * (macro['VIX'] / macro['VIX'].shift(20) - 1.0)
def sig_dvr():
    down = ret.clip(upper=0).rolling(20).std()
    return down / ret.rolling(120).std()
def sig_betacn10():
    r10 = close['CN10Y'].pct_change()
    return ret.rolling(60).cov(r10) / r10.rolling(60).var()
def sig_betachi():
    rhi = close['HSI'].pct_change()
    return ret.rolling(60).cov(rhi) / rhi.rolling(60).var()
def sig_corr10y():
    r10 = close['US10Y'].pct_change()
    return ret.rolling(60).corr(r10)
def sig_skew(): return -ret.rolling(20).skew()
def sig_vovchg():
    v = ret.rolling(20).std()
    return v.diff(20) / v.rolling(20).mean()
def sig_xaucop():
    cond = ((close['XAU'].pct_change(20) > 0) & (close['COPPER'].pct_change(20) > 0)).astype(float)
    return cond.mul(-1.0, axis=0)
def sig_volbeta():
    v = ret.rolling(20).std()
    spx_v = ret['SPX'].rolling(20).std()
    return v.rolling(60).cov(spx_v) / spx_v.rolling(60).var()
def sig_signewma():
    return (close / close.ewm(span=60).mean() - 1.0).apply(np.sign)

factors = {
    'mom_10d_skip5': sig_mom10,
    'mom_120d_skip5': sig_mom120,
    'vol_of_vol20x60': sig_vov,
    'low_vol_20d': sig_lowvol,
    'beta_vix_60d_neg': sig_betavix_neg,
    'vix_beta_cond_60x20': sig_vixcond,
    'down_vol_ratio_20x120': sig_dvr,
    'beta_cn10y_60d': sig_betacn10,
    'beta_chi_60d': sig_betachi,
    'corr_us10y_60d': sig_corr10y,
    'skew_20d_neg': sig_skew,
    'vol_of_vol_chg_20d': sig_vovchg,
    'xau_copper_cond_20d': sig_xaucop,
    'vol_beta_spx_60d': sig_volbeta,
    'sign_ewma_60d': sig_signewma,
}

print(f"Revalidation through {close.index[-1].date()} | rows={len(close)} | last5={list(close.index[-5:].astype(str))}", flush=True)
results = []
for name in sorted(factors):
    try:
        f = factors[name]().reindex(close.index)
        ic, n = vlib.rank_ic_series(f, fwd[5])
        ic = ic.dropna()
        s = vlib.summarize(ic, n.reindex(ic.index), name, fwd=fwd, factor_df=f, label='reval_20340831')
        dec = {}
        for h in [1,2,3,5,10,20]:
            ic_h, _ = vlib.rank_ic_series(f, fwd[h])
            ic_h = ic_h.dropna()
            dec[h] = float(ic_h.mean()) if len(ic_h) else np.nan
        s['decay_ic'] = dec
        s['pass_gate'] = abs(s['ic']) >= vlib.IC_TH and abs(s['icir']) >= vlib.ICIR_TH
        results.append(s)
        print(f"{name:24s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['hit']:.2f} n_dates={s['n_dates']:4d} "
              f"cov={s['coverage_asset_days']:.2f} r2020_22={s.get('r2020_22_ic',np.nan):+.3f} r2023_26={s.get('r2023_26_ic',np.nan):+.3f} "
              f"PASS={s['pass_gate']} decay10={dec[10]:+.3f}", flush=True)
    except Exception as e:
        print(f"{name:24s} ERROR: {type(e).__name__}: {e}", flush=True)
        results.append({'factor': name, 'error': str(e)})

json.dump(results, open('scripts/miner_1_20340831_revalidate_results.json', 'w'), indent=1, default=str)
print('\nSaved results.', flush=True)