import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=None
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:d=fn(s,5000)
  except:pass
  if d is not None and len(d):break
 if d is not None and len(d):
  z=d.copy();z.date=pd.to_datetime(z.date);P[s]=z.set_index('date').close.astype(float).rename(s)
px=pd.DataFrame(P).sort_index();r=px.pct_change()
down=(r.clip(upper=0).abs().mean(axis=1)/(r.abs().mean(axis=1)+1e-12)).rolling(20,min_periods=10).mean()
sig=-r.rolling(5,min_periods=5).sum()*(1+down);sig=sig.sub(sig.median(axis=1),axis=0)
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c):rows.append((dt,len(z),c))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');q=a.ic
print('assets',len(P),'dates',len(a),'median_n',a.n.median(),'coverage',a.n.sum()/(len(a)*15))
print('daily IC',q.mean(),'std',q.std(ddof=1),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
for h in [3,5,10,20]:
 y=px.pct_change(h).shift(-h);v=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c):v.append(c)
 ss=pd.Series(v);print('h',h,'IC',ss.mean(),'ICIR',ss.mean()/ss.std(ddof=1),'n',len(ss))
for nm,sl in [('2020-22',slice('2020','2022')),('2023-25',slice('2023','2025')),('2026-27',slice('2026','2027')),('2028-31',slice('2028','2031'))]:
 ss=a.loc[sl,'ic'].dropna();print(nm,len(ss),ss.mean(),ss.mean()/ss.std(ddof=1))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20310306_downside_asym_reversal_signal.csv',index=False)
