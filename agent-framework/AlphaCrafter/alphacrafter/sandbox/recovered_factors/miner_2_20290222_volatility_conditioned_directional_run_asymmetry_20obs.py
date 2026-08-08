"""One candidate: volatility-conditioned directional run asymmetry, through 2029-02-21.
The 20-day difference between longest positive and negative sign runs is scaled by
inverse 20-day realized volatility. It tests whether directional persistence is more
predictive when it emerges in a relatively orderly (low-volatility) path.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2029-02-21')
def get(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d.loc[:END,'close'].astype(float)
P=pd.DataFrame({a:get(a) for a in A}).sort_index(); r=P.pct_change()
def asym(x):
 q=np.sign(x.dropna().values); bestp=bestn=curp=curn=0
 for v in q:
  curp=curp+1 if v>0 else 0; curn=curn+1 if v<0 else 0
  bestp=max(bestp,curp);bestn=max(bestn,curn)
 return (bestp-bestn)/20
run=r.rolling(20,min_periods=15).apply(asym,raw=False)
vol=r.rolling(20,min_periods=15).std(ddof=1)
# Scaling cross-asset persistence by its own path variability avoids rewarding noisy runs.
F=run/(vol+1e-8); F=F.replace([np.inf,-np.inf],np.nan)
def metrics(h):
 fw=P.shift(-h)/P-1; vals=[]; ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),fw.loc[d].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.r.nunique()>1:
   vals.append((d,float(spearmanr(z.f,z.r).statistic)));ns.append(len(z))
 s=pd.Series(dict(vals),dtype=float);sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
for h in [1,5,10,20]:
 s,m=metrics(h);print('HORIZON',h,json.dumps(m,sort_keys=True))
 if h==10:
  for lab,mask in [('2020_2021',s.index.year<=2021),('2022_2023',s.index.year.isin([2022,2023])),('2024_2025',s.index.year.isin([2024,2025])),('2026_2028',s.index.year.isin([2026,2027,2028])),('2029',s.index.year==2029)]:
   q=s[mask];print('REGIME_10D',lab,json.dumps({'dates':len(q),'ic':float(q.mean()) if len(q) else None,'icir':float(q.mean()/q.std(ddof=1)) if len(q)>1 else None,'hit':float((q>0).mean()) if len(q) else None}))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
print('PANEL',json.dumps({'signal_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_names':float(F.notna().sum(axis=1).mean()),'rank_stability':float(np.mean(st)),'implied_turnover':float(1-np.mean(st))}))
F.to_pickle('scripts/miner_2_20290222_volatility_conditioned_directional_run_asymmetry_20obs_signal.pkl')
