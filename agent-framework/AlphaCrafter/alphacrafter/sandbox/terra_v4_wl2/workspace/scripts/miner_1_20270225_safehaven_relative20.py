import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 try:d=get_index_daily_data(s,days=5000)
 except FileNotFoundError:d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):P[s]=d.set_index(pd.to_datetime(d.date)).close
px=pd.DataFrame(P).sort_index(); r=px.pct_change(20); sig=r.sub(r.median(axis=1),axis=0).shift(1)
for h in [1,5,10]:
 f=px.pct_change(h).shift(-h); qs=[];ns=[];ds=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
 q=pd.Series(qs).dropna();print('h',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 for label,y0,y1 in [('2020-22',2020,2022),('2023-24',2023,2024),('2025-26',2025,2026),('2027',2027,2027)]:
  x=q[[d.year>=y0 and d.year<=y1 for d in ds]];print(label,round(x.mean(),6),len(x))
print('coverage',sig.notna().sum(axis=1).mean()/len(U),'active',sig.dropna(how='all').shape[0],'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=sig.stack().rename('signal').reset_index();out.columns=['date','asset','signal'];out.to_csv('../persistent/factor_signals_miner_1_20270225_safehaven_relative20.csv',index=False)
