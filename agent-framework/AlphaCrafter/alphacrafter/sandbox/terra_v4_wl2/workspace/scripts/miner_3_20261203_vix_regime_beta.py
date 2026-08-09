import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2026-12-03')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}
p=pd.concat(D,axis=1).sort_index();r=p.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut].reindex(r.index).ffill(); vr=vix.pct_change()
for w in [40,60,120]:
 b=pd.DataFrame(index=r.index,columns=U,dtype=float)
 for i in range(w,len(r)):
  z=vr.iloc[i-w:i]; vz=z.var()
  if vz>0:b.iloc[i]=r.iloc[i-w:i].apply(lambda q:q.cov(z)/vz)
 for mode in ['raw','regime']:
  f=-b.copy()
  if mode=='regime':
   # emphasize VIX-sensitive defensiveness only when VIX is rising over recent window
   regime=(vix/vix.shift(20)-1).clip(-1,1)
   f=f.mul(regime,axis=0)
  vals=[];dates=[];ns=[]
  for i in range(len(r)-1):
   q=pd.concat([f.iloc[i],r.iloc[i+1].rename('y')],axis=1).dropna()
   if len(q)>=8 and q.iloc[:,0].nunique()>1:vals.append(spearmanr(q.iloc[:,0],q.y).statistic);dates.append(r.index[i]);ns.append(len(q))
  a=np.array(vals); print('W',w,mode,'dates',len(a),'N',round(np.mean(ns),2),'cov',round(np.mean(ns)/15,3),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round(np.mean(a>0),4),'turn',round(np.nanmean(np.abs(f.rank(pct=True).diff()).mean(axis=1)),4))
  for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
   z=a[(pd.DatetimeIndex(dates).year>=lo)&(pd.DatetimeIndex(dates).year<=hi)]
   print(' reg',lo,hi,'IC',round(z.mean(),5),'ICIR',round(z.mean()/z.std(ddof=1),4),'n',len(z))
  if w==60 and mode=='regime':
   rows=[(dt,s,f.loc[dt,s]) for dt in f.index for s in U if pd.notna(f.loc[dt,s])]
   pd.DataFrame(rows,columns=['date','symbol','signal']).to_csv('../persistent/factor_signals_miner_3_20261203_vix_regime_beta.csv',index=False)
