import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x): D[s]=x.assign(date=pd.to_datetime(x.date)).set_index('date')['close'].astype(float)
P=pd.concat(D,axis=1).sort_index().ffill(); r=P.pct_change(); v=r.rolling(20,min_periods=15).std()*np.sqrt(20)
# Shock reversal: fade a lagged 5d move only when its magnitude exceeds one normalised volatility unit.
raw=-(P/P.shift(5)-1)/v.replace(0,np.nan)
F=raw.where(raw.abs()>=1.0,0.0).shift(1)
print('universe',len(D),'dates',len(P))
for h in [1,5,10,20]:
 R=P.shift(-h)/P-1; a=[]; ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d],R.loc[d]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): a.append(c);ns.append(len(z))
 q=pd.Series(a); print('H',h,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'dates',len(q),'avgN',round(np.mean(ns),2))
 if h==10:
  for n in [180,500,750]:
   z=q.tail(n); print('recent',n,'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'dates',len(z))
print('coverage',round(F.notna().mean().mean(),4),'active',round((F!=0).mean().mean(),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),4))
out=F.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20340807_shock_reversal_signal.csv',index=False); print('artifact',len(out))
