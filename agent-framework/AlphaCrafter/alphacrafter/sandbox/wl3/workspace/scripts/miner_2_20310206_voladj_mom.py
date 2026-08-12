import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=None
 for f in (get_stock_daily_data,get_index_daily_data):
  try:d=f(s,5000)
  except:pass
  if d is not None and len(d):break
 if d is not None and len(d):
  d.date=pd.to_datetime(d.date);P[s]=d.set_index('date').close.astype(float)
px=pd.concat(P,axis=1).sort_index();r=px.pct_change()
# risk-adjusted medium momentum: lagged 20d return divided by lagged 20d realized vol
sig=(r.rolling(20).sum()/r.rolling(20).std()).shift(1)
sig=sig.sub(sig.median(axis=1),axis=0)
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c):rows.append((dt,len(z),c))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');q=a.ic
print('dates',len(a),'median_n',a.n.median(),'coverage',a.n.sum()/(len(a)*15));print('daily',q.mean(),q.std(),q.mean()/q.std(),(q>0).mean())
for h in [3,5,10,20]:
 y=px.pct_change(h).shift(-h);v=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c):v.append(c)
 s=pd.Series(v);print(h,s.mean(),s.mean()/s.std(),len(s))
for nm,sl in [('2026-27',slice('2026','2027')),('2028-30',slice('2028','2030'))]:
 s=a.loc[sl,'ic'];print(nm,len(s),s.mean(),s.mean()/s.std())
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20310206_voladj_mom_signal.csv',index=False)
