"""miner_2 20271202: upside-concentration continuation validation."""
import os,json,glob
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-12-01')
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
P=pd.DataFrame({a:load(a) for a in A}).sort_index().loc[:END]; r=P.pct_change()
# Pre-specified continuation signal: fraction of 40-observation positive return magnitude concentrated in latest 10 observations.
F=r.clip(lower=0).rolling(10,min_periods=8).sum()/r.clip(lower=0).rolling(40,min_periods=28).sum()
def metric(h):
 R=P.shift(-h)/P-1; vals=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),R.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.r).statistic
   if np.isfinite(q): vals.append((dt,float(q)));ns.append(len(z))
 x=pd.Series(dict(vals),dtype=float);x.index=pd.to_datetime(x.index); sd=x.std(ddof=1)
 return x,{'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'ic_hit_ratio':float((x>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(x))),'ic_dates':len(x),'mean_valid_instruments_per_ic_date':float(np.mean(ns))}
M={}
for h in [1,5,10,20]:
 x,M[h]=metric(h);print('HORIZON',h,json.dumps(M[h],sort_keys=True))
x,_=metric(5)
for lab,mask in [('2020',x.index.year==2020),('2021_2022',x.index.year.isin([2021,2022])),('2023_2024',x.index.year.isin([2023,2024])),('2025_2027',x.index.year>=2025)]:
 y=x[mask];print('REGIME_5D',lab,'dates',len(y),'IC',float(y.mean()) if len(y) else None,'ICIR',float(y.mean()/y.std(ddof=1)) if len(y)>1 else None,'hit',float((y>0).mean()) if len(y) else None)
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8: st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
# Compare every current admitted signal artifact, excluding backups/rejected scripts.
paths=[p for p in glob.glob('scripts/*_signal.pkl') if 'miner_' in os.path.basename(p)]
E={};mx=0.; ok=True
for p in paths:
 n=os.path.basename(p).replace('_signal.pkl',''); L=pd.read_pickle(p);L.index=pd.to_datetime(L.index)
 L=L.reindex(index=F.index,columns=A);z=pd.concat([F.stack().rename('a'),L.stack().rename('b')],axis=1).dropna()
 q=float(spearmanr(z.a,z.b).statistic) if len(z)>=8 else None; E[n]={'rho':q,'common_signal_cells':len(z)}
 if q is None:ok=False
 else: mx=max(mx,abs(q))
 print('LIBRARY_CORR',n,'cells',len(z),'spearman',q)
print('FACTOR upside_concentration_continuation_10v40obs')
print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'mean_rank_stability_1d',float(np.mean(st)))
print('DECAY',json.dumps({str(k):v for k,v in M.items()},sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx if ok else None,'COMPLETE_EVIDENCE',ok,'EVIDENCE',json.dumps(E,sort_keys=True));F.to_pickle('scripts/miner_2_20271202_upside_concentration_continuation_10v40obs_signal.pkl')
