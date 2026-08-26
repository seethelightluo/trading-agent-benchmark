import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x): D[s]=x.assign(date=pd.to_datetime(x.date)).set_index('date')['close'].astype(float)
P=pd.concat(D,axis=1).sort_index().ffill(); daily=P.pct_change(); vol=P.pct_change().rolling(20,min_periods=15).std()*np.sqrt(20)
F=(-(P/P.shift(5)-1)/vol.replace(0,np.nan)).shift(1)
for h in [1,5,10,20]:
 R=P.shift(-h)/P-1; a=[]; ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d],R.loc[d]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): a.append(c);ns.append(len(z))
 q=pd.Series(a); print('H',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'dates',len(q),'avgN',np.mean(ns))
 if h==1:
  for n in [180,500,750]:
   z=q.tail(n);print('recent',n,z.mean(),z.mean()/z.std(ddof=1),len(z))
print('coverage',F.notna().mean().mean(),'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
out=F.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20340724_short_reversal_signal.csv',index=False);print('artifact',len(out))
