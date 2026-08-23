#!/usr/bin/env python3
"""Revalidate ALL library factors. Current date: 2027-08-12"""
import json,os,sys,math,glob,time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data,get_account_dict
np.seterr(all='ignore')

acct=get_account_dict(); wl=acct.get('watch_list',[])
print(f"Watchlist ({len(wl)}): {wl}")
idx={}
for iid in ['DXY','USDCNY','USDJPY','EURUSD','VIX']:
    d=get_index_daily_data(symbol=iid,days=2000)
    if d is not None and len(d)>60: idx[iid]=d
inst={}
for s in wl:
    d=get_stock_daily_data(symbol=s,days=2000)
    if d is not None and len(d)>60: inst[s]=d
print(f"Instruments ({len(inst)}): {list(inst.keys())}")
closes={s:inst[s].set_index('date')['close'] for s in inst}
cdf=pd.DataFrame(closes).sort_index().astype(float)
print(f"Close: {cdf.shape}, {cdf.index[0].date()} to {cdf.index[-1].date()}")
r=cdf.pct_change()
vix=idx['VIX'].set_index('date')['close'].pct_change() if 'VIX' in idx else None
dxy=idx['DXY'].set_index('date')['close'].pct_change() if 'DXY' in idx else None
cny=idx['USDCNY'].set_index('date')['close'].pct_change() if 'USDCNY' in idx else None
f10=cdf.shift(-10)/cdf-1

def ic_series(fv,fd,minv=8):
    ci=fv.index.intersection(fd.index); ic,n=[],[]
    for d in ci:
        u=fv.loc[d].dropna(); v=fd.loc[d].dropna()
        ok=u.index.intersection(v.index)
        if len(ok)<minv: continue
        a,b=u[ok].values,v[ok].values
        if np.std(a)>1e-12 and np.std(b)>1e-12:
            ic.append(pearsonr(a,b)[0]); n.append(len(ok))
    return np.array(ic),np.array(n)

def rpt(nm,fv,fd,lb='f10'):
    ic,n=ic_series(fv,fd)
    if len(ic)<4: print(f"  {nm:30s} SKIP ({len(ic)} dates)"); return None
    mi=np.mean(ic); ir=mi/np.std(ic) if np.std(ic)>1e-12 else 0; ht=np.mean(ic>0)
    print(f"  {nm:30s} dates={len(ic):4d} IC={mi:+.6f} ICIR={ir:+.6f} hit={ht:.3f} [{lb}]")
    return {'mean_ic':mi,'icir':ir,'hit_ratio':float(ht),'n_dates':len(ic)}

GATE_IC=0.007; GATE_IR=0.084
print(f"\n{'='*70}")
print(f"REVALIDATION ({cdf.index[-1].date()})")
print(f"Gates: |IC|>={GATE_IC} |ICIR|>={GATE_IR}")
print('='*70)
res={}

# 1: beta_VIX_60
print('\n--- beta_VIX_60 ---')
if vix is not None:
    t0=time.time()
    ci=cdf.index.intersection(vix.index)
    bv=pd.DataFrame(index=ci,columns=cdf.columns,dtype=float)
    for s in cdf.columns:
        for i in range(60,len(ci)):
            d=ci[i]; ir=cdf[s].pct_change().loc[ci[i-60]:d].dropna()
            vr=vix.loc[ci[i-60]:d].dropna()
            ok=ir.index.intersection(vr.index)
            if len(ok)>30: bv.loc[d,s]=np.cov(ir.loc[ok],vr.loc[ok])[0,1]/np.var(vr.loc[ok])
    res['beta_VIX_60']=rpt('beta_VIX_60',bv,f10)
    print(f"  t={time.time()-t0:.1f}s")

# 2: kaufman_eff_20d
print('\n--- kaufman_eff_20d ---')
t0=time.time()
kf=pd.DataFrame(index=cdf.index,columns=cdf.columns,dtype=float)
for s in cdf.columns:
    for i in range(20,len(cdf)):
        p=cdf[s].iloc[i-20:i+1].values
        d=abs(p[-1]-p[0]); v=np.sum(np.abs(np.diff(p)))
        kf.loc[cdf.index[i],s]=d/v if v>1e-12 else 0
res['kaufman_eff_20d']=rpt('kaufman_eff_20d',kf,f10)
print(f"  t={time.time()-t0:.1f}s")

# 3: mom_120d_skip5
print('\n--- mom_120d_skip5 ---')
res['mom_120d_skip5']=rpt('mom_120d_skip5',cdf.pct_change(120).shift(5),f10)

# 4: bb_width_20d
print('\n--- bb_width_20d ---')
bb=(2*cdf.rolling(20).std())/cdf.rolling(20).mean()
res['bb_width_20d']=rpt('bb_width_20d',bb.shift(1),f10)

# 5: cny_beta_60
print('\n--- cny_beta_60 ---')
if cny is not None:
    t0=time.time()
    ci=cdf.index.intersection(cny.index)
    bc=pd.DataFrame(index=ci,columns=cdf.columns,dtype=float)
    for s in cdf.columns:
        for i in range(60,len(ci)):
            d=ci[i]; ir=cdf[s].pct_change().loc[ci[i-60]:d].dropna()
            cr=cny.loc[ci[i-60]:d].dropna()
            ok=ir.index.intersection(cr.index)
            if len(ok)>30: bc.loc[d,s]=np.cov(ir.loc[ok],cr.loc[ok])[0,1]/np.var(cr.loc[ok])
    res['cny_beta_60']=rpt('cny_beta_60',bc,f10)
    print(f"  t={time.time()-t0:.1f}s")

# 6: vol_z_20d
print('\n--- vol_z_20d ---')
v20=r.rolling(20).std(); vm=v20.rolling(120).mean(); vs=v20.rolling(120).std()
vz=(v20-vm)/vs
res['vol_z_20d']=rpt('vol_z_20d',vz.shift(1),f10)

# 7: ac1_120d (autocorrelation)
print('\n--- ac1_120d ---')
ac1=pd.DataFrame(index=cdf.index,columns=cdf.columns,dtype=float)
for s in cdf.columns:
    for i in range(120,len(cdf)):
        sr=cdf[s].iloc[i-120:i+1].values; sr_ret=(sr[1:]-sr[:-1])/sr[:-1]
        if np.std(sr_ret)<1e-12: continue
        ac1.loc[cdf.index[i],s]=np.corrcoef(sr_ret[:-1],sr_ret[1:])[0,1]
res['ac1_120d']=rpt('ac1_120d',ac1,f10)

# 8: dxy_corr_change_20_60
print('\n--- dxy_corr_change_20_60 ---')
if dxy is not None:
    ci=cdf.index.intersection(dxy.index)
    dcc=pd.DataFrame(index=ci,columns=cdf.columns,dtype=float)
    for s in cdf.columns:
        for i in range(60,len(ci)):
            d=ci[i]; ir=cdf[s].pct_change().loc[ci