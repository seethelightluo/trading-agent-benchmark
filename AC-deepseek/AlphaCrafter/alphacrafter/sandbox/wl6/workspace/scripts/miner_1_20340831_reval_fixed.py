"""miner_1 2034-08-31: fixed revalidation of factor library + ensemble through visible_through."""
import json, importlib.util
import numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location("fl", "scripts/miner_1_20281102_fastlib.py")
fl = importlib.util.module_from_spec(spec); spec.loader.exec_module(fl)

close = fl.load_close_panel()
ret = close.pct_change()
macro = fl.load_macro_panel()
fwd = fl.forward_returns(close, horizons=(1, 2, 3, 5, 10, 20))

def beta_to_series(asset_ret, benchmark_ret):
    """Per-asset rolling beta: asset_ret (DataFrame), benchmark_ret (Series)."""
    bench = benchmark_ret.reindex(asset_ret.index)
    out = pd.DataFrame(index=asset_ret.index, columns=asset_ret.columns, dtype=float)
    for c in asset_ret.columns:
        a = asset_ret[c]
        cov = a.rolling(60).cov(bench)
        var = bench.rolling(60).var()
        out[c] = cov / var
    return out

def sig_mom10(): return close.shift(5)/close.shift(15)-1.0
def sig_mom120(): return close.shift(5)/close.shift(125)-1.0
def sig_vov(): return ret.rolling(20).std().rolling(60).std()
def sig_lowvol(): return -ret.rolling(20).std()
def sig_betavix_neg():
    return -beta_to_series(ret, macro['VIX'].pct_change())
def sig_vixcond():
    beta = beta_to_series(ret, macro['VIX'].pct_change())
    return -beta*(macro['VIX']/macro['VIX'].shift(20)-1.0)
def sig_dvr():
    down=ret.clip(upper=0).rolling(20).std(); return down/ret.rolling(120).std()
def sig_betacn10():
    return beta_to_series(ret, close['CN10Y'].pct_change())
def sig_betachi():
    return beta_to_series(ret, close['HSI'].pct_change())
def sig_corr10y():
    return ret.rolling(60).corr(close['US10Y'].pct_change())
def sig_skew(): return -ret.rolling(20).skew()
def sig_vovchg():
    v=ret.rolling(20).std(); return v.diff(20)/v.rolling(20).mean()
def sig_xaucop():
    cond=((close['XAU'].pct_change(20)>0)&(close['COPPER'].pct_change(20)>0)).astype(float)
    return -cond
def sig_volbeta():
    v=ret.rolling(20).std(); spx_v=ret['SPX'].rolling(20).std()
    return beta_to_series(v, spx_v)
def sig_signewma():
    return (close/close.ewm(span=60).mean()-1.0).apply(np.sign)

factors = {
 'mom_10d_skip5':sig_mom10,'mom_120d_skip5':sig_mom120,'vol_of_vol20x60':sig_vov,
 'low_vol_20d':sig_lowvol,'beta_vix_60d_neg':sig_betavix_neg,'vix_beta_cond_60x20':sig_vixcond,
 'down_vol_ratio_20x120':sig_dvr,'beta_cn10y_60d':sig_betacn10,'beta_chi_60d':sig_betachi,
 'corr_us10y_60d':sig_corr10y,'skew_20d_neg':sig_skew,'vol_of_vol_chg_20d':sig_vovchg,
 'xau_copper_cond_20d':sig_xaucop,'vol_beta_spx_60d':sig_volbeta,'sign_ewma_60d':sig_signewma,
}

print(f"through {close.index[-1].date()} rows={len(close)}", flush=True)
results=[]
for name in sorted(factors):
    try:
        f=factors[name]().reindex(close.index)
        ic,n=fl.rank_ic_series(f, fwd[5]); ic=ic.dropna()
        s=fl.summarize(ic,n.reindex(ic.index),name,fwd=fwd,factor_df=f,label='reval_fixed_20340831')
        s['pass_gate']=abs(s['ic'])>=fl.IC_TH and abs(s['icir'])>=fl.ICIR_TH
        # recent-window IC
        sub=ic[(ic.index>='2032-01-01')]
        s['r2032_ic']=float(sub.mean()) if len(sub) else np.nan
        s['r2032_n']=int(len(sub))
        results.append(s)
        print(f"{name:22s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['hit']:.2f} n={s['n_dates']:4d} "
              f"cov={s['coverage_asset_days']:.2f} r20_22={s.get('r2020_22_ic',np.nan):+.3f} "
              f"r23_26={s.get('r2023_26_ic',np.nan):+.3f} r32={s['r2032_ic']:+.3f}({s['r2032_n']}) "
              f"turn={s.get('turnover_rank_abs',np.nan):.2f} d5={s['decay_ic'].get('5',np.nan):+.3f} "
              f"d10={s['decay_ic'].get('10',np.nan):+.3f} PASS={s['pass_gate']}", flush=True)
    except Exception as e:
        print(f"{name:22s} ERROR {type(e).__name__}:{e}", flush=True)
        results.append({'factor':name,'error':str(e)})

json.dump(results,open('scripts/miner_1_20340831_reval_fixed_results.json','w'),indent=1,default=str)
print('saved')