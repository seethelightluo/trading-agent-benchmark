import os, json
import numpy as np
import pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2031-08-21')
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d.date); P[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(P).sort_index().loc[:cutoff]; r5=P/P.shift(5)-1; D=r5.std(axis=1).where(r5.notna().sum(axis=1)>=8); th=D.rolling(252,min_periods=100).quantile(.60)
sig=(-r5).where(D.gt(th),np.nan); fwd=P.shift(-1)/P-1; rows=[]; art=[]
for dt in P.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
  for s in z.index: art.append((dt,s,float(sig.loc[dt,s])))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); mu=q.ic.mean(); sd=q.ic.std(ddof=1)
print(json.dumps({'dates':len(q),'start':str(q.index.min().date()),'end':str(q.index.max().date()),'avg_instruments':q.n.mean(),'daily_ic':mu,'daily_icir':mu/sd,'hit_ratio':(q.ic>0).mean(),'coverage':q.n.mean()/15,'turnover':sig.rank(pct=True).diff().abs().mean(axis=1).mean()},indent=2))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2031-08-21')]:
 t=q.loc[a:b]
 if len(t): print(a,b,len(t),t.ic.mean(),t.ic.mean()/t.ic.std(ddof=1),(t.ic>0).mean())
pd.DataFrame(art,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20310821_dispersion_p60_reversal_signal.csv',index=False)
