import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:get_stock_daily_data(s,5000).set_index('date')['close'] for s in U}).sort_index()
# Lagged multi-horizon trend alignment: 20d trend plus slower 60d confirmation, scaled by 20d vol.
r20=P.shift(1)/P.shift(21)-1; r60=P.shift(1)/P.shift(61)-1
v=P.pct_change().rolling(20,min_periods=15).std().shift(1)
F=((r20+0.5*r60)/(v+1e-6)).clip(-8,8); F=F.sub(F.median(axis=1),axis=0)
def evalh(h):
 rows=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
 return pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
R=evalh(10); q=R.ic
print('dates',len(R),'range',R.index.min(),R.index.max(),'avg_n',R.n.mean(),'coverage',R.n.mean()/15)
print('IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [5,20]:
 x=evalh(h).ic; print('decay',h,len(x),x.mean(),x.mean()/x.std(ddof=1))
for n in [180,360]:
 x=q.tail(n); print('recent',n,x.mean(),x.mean()/x.std(ddof=1))
print('last',P.index.max())
F.to_csv('scripts/miner_2_20300516_trend_alignment_signal.csv');R.to_csv('scripts/miner_2_20300516_trend_alignment_ic.csv')
