import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);v=v.drop_duplicates('date').set_index('date').close.astype(float)
P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);P[s]=d.drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill(); r=np.log(p).diff(); vol=r.rolling(20,min_periods=15).std()
# Shock-amplified short reversal, with slow trend anchor to avoid a sparse/constant signal.
rev=-np.log(p).diff(5)/(vol*np.sqrt(5)+1e-9); trend=np.log(p).diff(40)/(vol*np.sqrt(40)+1e-9)
vl=np.log(v); vz=(vl-vl.rolling(120,min_periods=60).mean())/(vl.rolling(120,min_periods=60).std()+1e-9); shock=vz.clip(lower=0,upper=2)
raw=rev*(1+0.5*shock.reindex(p.index).fillna(0))+0.20*trend
sig=raw.shift(1).rank(axis=1,pct=True); rows=[]
for dt in sig.index:
 i=p.index.get_loc(dt)
 for h in [1,5,10,20]:
  if i+h>=len(p):continue
  z=pd.concat([sig.loc[dt].rename('x'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,z.x.corr(z.y,method='spearman'),len(z)))
o=pd.DataFrame(rows,columns=['date','h','ic','n']);o.date=pd.to_datetime(o.date);print('dates',o.date.nunique(),'assets',p.shape[1],'avgN',o.groupby('date').n.first().mean(),'coverage',sig.notna().mean().mean(),'turnover',sig.diff().abs().mean().mean())
for h in [1,5,10,20]:
 q=o[o.h==h].groupby('date').ic.first().dropna();print('h',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for yr,g in o[o.h==10].groupby(o[o.h==10].date.dt.year):
 q=g.groupby('date').ic.first().dropna();print('year',yr,'IC %.6f ICIR %.6f n=%d'%(q.mean(),q.mean()/q.std(ddof=1),len(q)))
sig.to_csv('scripts/miner_1_20310403_shock_reversal_trend_signal.csv')
