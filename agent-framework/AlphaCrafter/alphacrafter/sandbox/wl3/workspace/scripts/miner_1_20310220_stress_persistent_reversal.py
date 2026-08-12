import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 d=None
 for f in (get_stock_daily_data,get_index_daily_data):
  try:d=f(s,5000)
  except:pass
  if d is not None and len(d):break
 if d is not None and len(d):
  d.date=pd.to_datetime(d.date);P[s]=d.set_index('date').close.astype(float).rename(s)
px=pd.concat(P,axis=1).sort_index();r=px.pct_change();vix=pd.read_csv('../persistent/index_data/VIX.csv');vix.date=pd.to_datetime(vix.date);v=vix.set_index('date')['close'].reindex(px.index).ffill();stress=(v>v.rolling(60,min_periods=40).median())&(v.diff(5)>0); active=stress.rolling(3,min_periods=3).sum()>=2
rv=r.rolling(20,min_periods=15).std();x=-(r.rolling(5,min_periods=5).sum().sub(r.rolling(5,min_periods=5).sum().median(axis=1),axis=0))/rv;x=x.where(active,0)
rows=[]
for dt in x.index:
 z=pd.concat([x.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c):rows.append((dt,len(z),c))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');q=a.ic;print('dates',len(a),'median_n',a.n.median(),'coverage',a.n.sum()/(len(a)*15));print('daily',q.mean(),q.std(),q.mean()/q.std(),(q>0).mean())
for h in [3,5,10]:
 y=px.pct_change(h).shift(-h);w=[]
 for dt in x.index:
  z=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c):w.append(c)
 s=pd.Series(w);print('h',h,'IC',s.mean(),'ICIR',s.mean()/s.std(),'n',len(s))
for nm,sl in [('2026-27',slice('2026','2027')),('2028-30',slice('2028','2030'))]:
 s=a.loc[sl,'ic'].dropna();print(nm,len(s),s.mean(),s.mean()/s.std())
out=x.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20310220_stress_persistent_reversal_signal.csv',index=False)
