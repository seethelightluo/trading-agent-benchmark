import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2034-04-25')
px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']); px[s]=d.drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index().loc[:cut]; r=p.pct_change(); vol=r.rolling(40,min_periods=25).std()*np.sqrt(20)
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close.astype(float).reindex(p.index).ffill()
# Dollar-strength regime: continuation when DXY rises, reversal otherwise.
reg=np.where(dxy.pct_change(20)>0,1.0,-1.0); sig=p.pct_change(20).div(vol).mul(reg,axis=0); fwd=p.shift(-10)/p-1
ics=[]; ns=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
ics=pd.Series(ics).dropna(); print('dates',len(ics),'meanN',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,4),'IC10',round(ics.mean(),6),'ICIR_ann',round(ics.mean()/ics.std()*np.sqrt(252),6),'hit',round((ics>0).mean(),4))
for h in [5,20]:
 vals=[]; ff=p.shift(-h)/p-1
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,round(np.nanmean(vals),6),len(vals))
for a,b in [('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-04-25')]:
 # recompute date keyed for regime reporting
 q=[]
 for dt in sig.loc[a:b].index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('regime',a[:4],round(np.nanmean(q),6),len(q))
sig.index.name='date'; sig.to_csv('scripts/miner_1_20340427_dxy_conditioned_continuation_signal.csv')
print('turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
