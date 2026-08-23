"""miner_3 2034-08-31: efficient revalidation of effective library + quick candidate screen.
Truncated to visible_through 2034-08-30. Subsampled every 2nd trading day for IC series speed."""
import json, numpy as np, pandas as pd
import importlib.util
spec = importlib.util.spec_from_file_location("vlib", "scripts/miner_1_20260730_validation_lib.py")
vlib = importlib.util.module_from_spec(spec); spec.loader.exec_module(vlib)

close = vlib.load_close_panel()
ret = close.pct_change()
macro = vlib.load_macro_panel()
fwd = vlib.forward_returns(close, horizons=(1,2,3,5,10,20))
vixr = macro['VIX'].pct_change()
print(f"rows={len(close)} last={close.index[-1].date()} cols={close.shape[1]}", flush=True)

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
    return ret.clip(upper=0).rolling(20).std()/ret.rolling(120).std()
def sig_betacn10():
    r10=close['CN10Y'].pct_change(); return ret.rolling(60).cov(r10)/r10.rolling(60).var()
def sig_betachi():
    rhi=close['HSI'].pct_change(); return ret.rolling(60).cov(rhi)/rhi.rolling(60).var()
def sig_corr10y():
    r10=close['US10Y'].pct_change(); return ret.rolling(60).corr(r10)
def sig_skew(): return -ret.rolling(20).skew()
def sig_vovchg():
    v=ret.rolling(20).std(); return v.diff(20)/v.rolling(20).mean()
def sig_xaucop():
    cond=((close['XAU'].pct_change(20)>0)&(close['COPPER'].pct_change(20)>0)).astype(float)
    return cond.mul(-1.0,axis=0)
def sig_volbeta():
    v=ret.rolling(20).std(); spxv=ret['SPX'].rolling(20).std()
    return v.rolling(60).cov(spxv)/spxv.rolling(60).var()
def sig_signewma():
    return (close/close.ewm(span=60).mean()-1.0).apply(np.sign)

factors = {
 'mom_10d_skip5':sig_mom10,'mom_120d_skip5':sig_mom120,'vol_of_vol20x60':sig_vov,
 'low_vol_20d':sig_lowvol,'beta_vix_60d_neg':sig_betavix_neg,'vix_beta_cond_60x20':sig_vixcond,
 'down_vol_ratio_20x120':sig_dvr,'beta_cn10y_60d':sig_betacn10,'beta_chi_60d':sig_betachi,
 'corr_us10y_60d':sig_corr10y,'skew_20d_neg':sig_skew,'vol_of_vol_chg_20d':sig_vovchg,
 'xau_copper_cond_20d':sig_xaucop,'vol_beta_spx_60d':sig_volbeta,'sign_ewma_60d':sig_signewma,
}

def fast_rank_ic(f, fwdh, stride=2):
    rows=[]
    idx=f.index
    for i in range(0, len(idx), stride):
        row=f.iloc[i]; fr=fwdh.iloc[i]
        m=row.notna() & fr.notna()
        if m.sum()<8: continue
        ic=row[m].rank().corr(fr[m].rank())
        if np.isfinite(ic): rows.append(ic)
    return np.array(rows)

res=[]
for name in sorted(factors):
    try:
        f=factors[name]().reindex(close.index)
        out={'factor':name}
        for h in [1,2,3,5,10,20]:
            ic=fast_rank_ic(f, fwd[h])
            out[f'ic{h}']=float(ic.mean()) if len(ic) else np.nan
            if h==5:
                out['icir5']=float(ic.mean()/ic.std(ddof=1)) if len(ic)>1 and ic.std(ddof=1)>0 else 0.0
                out['hit5']=float((ic>0).mean()); out['n5']=int(len(ic))
        out['cov']=float(f.notna().mean().mean())
        out['pass']=abs(out['ic5'])>=0.0070 and abs(out['icir5'])>=0.0840
        res.append(out)
        print(f"{name:22s} ic1={out['ic1']:+.4f} ic5={out['ic5']:+.4f} icir5={out['icir5']:+.3f} hit={out['hit5']:.2f} ic10={out['ic10']:+.4f} ic20={out['ic20']:+.4f} PASS={out['pass']}", flush=True)
    except Exception as e:
        print(f"{name} ERROR {e}", flush=True)
json.dump(res, open('scripts/miner_3_20340831_reval.json','w'), indent=1)
print('saved', flush=True)