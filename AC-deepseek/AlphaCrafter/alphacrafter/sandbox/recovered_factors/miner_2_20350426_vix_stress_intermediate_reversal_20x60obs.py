"""Miner_2 single-idea research: VIX-stress-conditioned intermediate cross-asset reversal, cutoff 2035-04-25.
The macro VIX state is observation-only; the resulting score applies solely to the 15 tradable assets.
"""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2035-04-25')
def px(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d.close.astype(float)
def ix(a):
 d=pd.read_csv('../persistent/index_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d.close.astype(float)
C=pd.DataFrame({a:px(a) for a in A}).loc[:END]
v=ix('VIX').reindex(C.index).ffill()
# On VIX levels above their 60-day mean, favor 20-day losers (stress reversal); outside stress values are missing, not neutral.
stress=(v/v.rolling(60,min_periods=45).mean()-1).clip(lower=0)
F=-C.pct_change(20).mul(stress.where(stress>0),axis=0)
def metrics(h):
 y=C.shift(-h).div(C)-1; vals=[]; cnt=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   r=spearmanr(z.f,z.y).statistic
   if np.isfinite(r): vals.append((d,r));cnt.append(len(z))
 s=pd.Series(dict(vals),dtype=float); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(cnt))}
for h in [1,5,10,20]:
 _,m=metrics(h); print('HORIZON',h,json.dumps(m,sort_keys=True))
one,_=metrics(1)
for n,years in [('2020_2022',[2020,2021,2022]),('2023_2025',[2023,2024,2025]),('2026_2028',[2026,2027,2028]),('2029_2031',[2029,2030,2031]),('2032_2034',[2032,2033,2034]),('2035',[2035])]:
 s=one[one.index.year.isin(years)];print('REGIME',n,'DATES',len(s),'IC',float(s.mean()) if len(s) else None,'ICIR',float(s.mean()/s.std(ddof=1)) if len(s)>1 else None,'HIT',float((s>0).mean()) if len(s) else None)
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p))
  if j.get('validation',{}).get('status')=='EFFECTIVE': active.append(j['factor_id'])
 except Exception: pass
signals=glob.glob('scripts/*_signal.pkl'); evidence={}; mx=0.; most=None; complete=True
for fid in active:
 matches=[p for p in signals if fid in os.path.basename(p)]
 if not matches: evidence[fid]=None;complete=False;continue
 try:
  L=pd.read_pickle(max(matches,key=os.path.getmtime)).reindex(index=F.index,columns=A)
  z=pd.concat([F.stack().rename('f'),L.stack().rename('l')],axis=1).dropna()
  r=spearmanr(z.f,z.l).statistic if len(z)>=8 and z.f.nunique()>1 and z.l.nunique()>1 else np.nan
 except Exception:r=np.nan
 evidence[fid]=float(r) if np.isfinite(r) else None
 if not np.isfinite(r):complete=False
 elif abs(r)>mx:mx=abs(float(r));most=fid
print('PANEL_DATES',len(F),'UNIVERSE',len(A),'COVERAGE',float(F.notna().mean().mean()),'MEAN_NAMES',float(F.notna().sum(axis=1).mean()),'STABILITY',float(np.nanmean(st)),'TURNOVER',float(1-np.nanmean(st)))
print('MAXCORR',mx,'MOST',most,'COMPLETE',complete,'COMPARED',len(active));print('EVIDENCE',json.dumps(evidence,sort_keys=True))
F.to_pickle('scripts/miner_2_20350426_vix_stress_intermediate_reversal_20x60obs_signal.pkl')
"""
# remove accidental dangling string
