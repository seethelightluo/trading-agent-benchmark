import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is None or len(x)<100: x=get_index_daily_data(s,days=4000)
 if x is not None:
  x=x.set_index('date'); D[s]=x[['open','close','high','low','volume']].astype(float)
# Asset-specific candle pressure reversal: reverse recent signed close-location pressure,
# scaled by recent volatility and liquidity-normalized volume surprise.
P=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index().ffill(); r=P.pct_change()
clv={s:((d.close-d.low)-(d.high-d.close))/(d.high-d.low).replace(0,np.nan) for s,d in D.items()}
C=pd.DataFrame(clv).reindex(P.index).ffill();
vol=pd.DataFrame({s:d.volume for s,d in D.items()}).reindex(P.index)
vs=np.log1p(vol).rolling(20,min_periods=10).mean(); vstd=np.log1p(vol).rolling(60,min_periods=30).std()
pressure=(C.rolling(3,min_periods=3).mean()*(1+0.25*((np.log1p(vol)-vs)/vstd).clip(-2,2)))
risk=r.rolling(20,min_periods=15).std()
f=-pressure/(risk*np.sqrt(3)); f=f.sub(f.median(axis=1),axis=0)
rows=[]
for i in range(len(P)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); q=a.ic
print('dates',len(q),'avgN',a.n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for nm,m in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-30',a.index>='2026-01-01')]:
 z=a.loc[m].ic; print(nm,len(z),z.mean(),z.mean()/z.std(ddof=1))
for k in [3,5,10]:
 y=P.pct_change().rolling(k).sum().shift(-k+1); o=[]
 for i in range(len(P)-k):
  z=pd.concat([f.iloc[i],y.iloc[i]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:o.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',k,'IC',np.nanmean(o),'n',len(o))
f.to_csv('scripts/miner_2_20301031_clv_pressure_signal.csv')
