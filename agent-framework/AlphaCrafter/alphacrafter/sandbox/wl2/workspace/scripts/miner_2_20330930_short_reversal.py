import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=4500)
 if d is None or len(d)<200:d=get_index_daily_data(s,days=4500)
 if d is not None:D[s]=d.set_index('date')['close'].astype(float)
C=pd.concat(D,axis=1).sort_index().ffill(); R=C.pct_change(); v=R.rolling(20,min_periods=15).std()
# Lagged 3-day residual reversal, volatility normalized; signal is known only at close t and predicts t+1 onward.
f=-(R.rolling(3).sum().shift(1).sub(R.rolling(3).sum().shift(1).mean(axis=1),axis=0))/v
f=f.replace([np.inf,-np.inf],np.nan)
def ev(h):
 a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],(C.shift(-h)/C-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]));ns.append(len(z))
 q=pd.Series(a).dropna();return q,np.mean(ns)
for h in [1,3,5,10,20]:
 q,n=ev(h);print('h',h,'dates',len(q),'avg_n',round(n,3),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==10:
  for name,s in [('2020-25',q.index<2171),('recent',q.index>=len(q)-1000)]: pass
print('coverage',C.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20330930_short_reversal_signal.csv',index=False)
