import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,2400)
 if x is None or len(x)<100:x=get_index_daily_data(s,2400)
 if x is not None and len(x):D[s]=x.set_index('date')
df=pd.concat({s:v for s,v in D.items()},axis=1).sort_index().ffill(); close=df.xs('close',axis=1,level=1); high=df.xs('high',axis=1,level=1); low=df.xs('low',axis=1,level=1)
r=close.pct_change(); atr=(high-low).rolling(14,min_periods=10).mean()/close
# range-location reversal: fade statistically extreme close location, scaled by range volatility
hi=close.rolling(12,min_periods=8).max(); lo=close.rolling(12,min_periods=8).min(); loc=(close-lo)/(hi-lo)
f=((0.5-loc)*2/(atr+1e-8)).shift(1)
rows=[]
for i in range(len(close)-10):
 for h in [1,5,10]:
  z=pd.concat([f.iloc[i],close.iloc[i+h]/close.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:rows.append((close.index[i],h,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
a=pd.DataFrame(rows,columns=['date','h','n','ic']).dropna()
for h in [1,5,10]:
 z=a[a.h==h].ic;print('h',h,'dates',len(z),'avgN',a[a.h==h].n.mean(),'IC',z.mean(),'ICIR_raw',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'ICIR_sqrtN',z.mean()/z.std(ddof=1)*np.sqrt(len(z)))
print('assets',len(D),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'last',close.index[-1])
for name,mask in [('2020-22',a.date<'2023-01-01'),('2023-25',(a.date>='2023-01-01')&(a.date<'2026-01-01')),('2026-30',a.date>='2026-01-01')]:
 z=a[(a.h==1)&mask].ic;print(name,len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1))
f.to_csv('scripts/miner_2_20300418_range_location_signal.csv')
