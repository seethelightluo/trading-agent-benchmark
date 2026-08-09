import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:get_stock_daily_data(a,days=4000) for a in U}; C=pd.DataFrame({a:d.set_index('date').close.astype(float) for a,d in D.items() if d is not None}).sort_index(); H=pd.DataFrame({a:d.set_index('date').high.astype(float) for a,d in D.items() if d is not None}).reindex(C.index); L=pd.DataFrame({a:d.set_index('date').low.astype(float) for a,d in D.items() if d is not None}).reindex(C.index)
# Distance from rolling midpoint, normalized by range: mean location in 20d channel.
hi=H.rolling(20,min_periods=15).max(); lo=L.rolling(20,min_periods=15).min(); f=((C-lo)/(hi-lo+1e-12)-.5).rolling(5,min_periods=3).mean()
f=f.clip(f.quantile(.05,axis=1),f.quantile(.95,axis=1),axis=0)
for h in [1,3,5,10]:
 y=C.shift(-h)/C-1; z=[];ns=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'));ns.append(len(q))
 s=pd.Series(z).dropna();print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
print('coverage',round(f.notna().mean().mean(),4),'turn',round(f.rank(pct=True,axis=1).diff().abs().mean(axis=1).mean(),4))
f.stack().rename('signal').to_csv('../persistent/factor_signals_miner_2_20270225_channel_location.csv')
