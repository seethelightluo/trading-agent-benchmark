import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:get_stock_daily_data(s,5000).set_index('date')['close'] for s in U}).sort_index()
r20=P.shift(1)/P.shift(21)-1;r60=P.shift(1)/P.shift(61)-1
# Relative trend persistence: centered 20d momentum plus 0.35 slower 60d confirmation.
F=r20.sub(r20.mean(axis=1),axis=0)+.35*r60.sub(r60.mean(axis=1),axis=0)
def ev(h):
 a=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:a.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
 return pd.DataFrame(a,columns=['date','n','ic']).set_index('date')
X=ev(10);q=X.ic
print('dates',len(X),'range',X.index.min(),X.index.max(),'avg_n',X.n.mean(),'coverage',X.n.mean()/15)
print('IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [5,20]:
 z=ev(h).ic;print('decay',h,len(z),z.mean(),z.mean()/z.std(ddof=1))
for n in [180,360]:
 z=q.tail(n);print('recent',n,z.mean(),z.mean()/z.std(ddof=1))
print('last',P.index.max());F.to_csv('scripts/miner_2_20300516_relative_trend_persistence_signal.csv');X.to_csv('scripts/miner_2_20300516_relative_trend_persistence_ic.csv')
