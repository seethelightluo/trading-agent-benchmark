import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=None
 for f in (get_stock_daily_data,get_index_daily_data):
  try:d=f(s,5000)
  except Exception:pass
  if d is not None and len(d):break
 if d is not None and len(d):
  x=d.copy();x.date=pd.to_datetime(x.date);P[s]=x.set_index('date').close.astype(float).rename(s)
px=pd.concat(P,axis=1).sort_index(); r=px.pct_change(); m=r.mean(axis=1)
# 20-day market-beta residual return, with beta estimated only through t-1.
sig=pd.DataFrame(index=px.index,columns=px.columns,dtype=float)
for s in px:
 beta=r[s].rolling(60,min_periods=40).cov(m)/m.rolling(60,min_periods=40).var()
 resid=r[s]-beta.shift(1)*m
 vol=resid.rolling(20,min_periods=15).std()
 sig[s]=(resid.rolling(10,min_periods=10).sum()/vol.replace(0,np.nan)).shift(0)
# cross-sectional demean; forward returns start after signal date
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c):rows.append((dt,len(z),c))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');q=a.ic
print('dates',len(a),'median_n',a.n.median(),'coverage',a.n.sum()/(len(a)*15))
print('daily IC',q.mean(),'std',q.std(),'ICIR',q.mean()/q.std(),'hit',(q>0).mean())
for h in [3,5,10,20]:
 y=px.pct_change(h).shift(-h);v=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c):v.append(c)
 ss=pd.Series(v);print('h',h,'IC',ss.mean(),'ICIR',ss.mean()/ss.std(),'n',len(ss))
for nm,sl in [('2020-22',slice('2020','2022')),('2023-25',slice('2023','2025')),('2026-27',slice('2026','2027')),('2028-30',slice('2028','2030'))]:
 ss=a.loc[sl,'ic'].dropna();print(nm,len(ss),ss.mean(),ss.mean()/ss.std())
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20310220_beta_residual_momentum_signal.csv',index=False)
