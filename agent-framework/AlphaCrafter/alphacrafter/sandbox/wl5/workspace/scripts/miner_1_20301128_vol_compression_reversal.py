import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
px=pd.DataFrame(C).sort_index(); r=px.pct_change(); vol20=r.rolling(20).std(); vol60=r.rolling(60).std()
base=-r.rolling(20).sum()*(vol60/vol20).clip(0.25,4); fwd=px.shift(-10)/px-1
rows=[]
for dt in base.index:
 z=pd.concat([base.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(a),'mean_n',a.n.mean(),'coverage',a.n.mean()/15); print('IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(),'hit',(a.ic>0).mean())
for p in [('2020','2024'),('2025','2027'),('2028','2029'),('2030','2030')]:
 q=a.loc[p[0]:p[1],'ic']; print(p,len(q),q.mean(),q.mean()/q.std())
for h in [5,10,20]:
 fw=px.shift(-h)/px-1; rr=[]
 for dt in base.index:
  z=pd.concat([base.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 q=pd.Series(rr).dropna(); print('h',h,'IC',q.mean(),'ICIR',q.mean()/q.std())
print('turnover',base.rank(axis=1,pct=True).diff().abs().mean().mean()); base.to_csv('scripts/miner_1_20301128_vol_compression_reversal_signal.csv')
