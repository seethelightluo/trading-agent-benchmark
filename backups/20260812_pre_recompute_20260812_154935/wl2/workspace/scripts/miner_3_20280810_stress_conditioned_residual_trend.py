import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<120:d=get_index_daily_data(s,days=3000)
 if d is not None:D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').sort_index().close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); print('p finite',p.notna().sum().sum())
ret=np.log(p).diff(20);med=ret.median(axis=1);stress=(-med).clip(lower=0).rolling(5,min_periods=1).mean(); base=ret.sub(med,axis=0); f=base.mul(1+2*stress,axis=0).shift(1)
print('finite f',f.notna().sum().sum(),'rows',f.dropna(how='all').shape)
for h in [1,3,5,10]:
 y=np.log(p).shift(-h)-np.log(p);a=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q):a.append(q);ns.append(len(z))
 a=np.asarray(a);print('h',h,'dates',len(a),'N',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6))
f.to_csv('scripts/miner_3_20280810_stress_conditioned_residual_trend_signal.csv')
