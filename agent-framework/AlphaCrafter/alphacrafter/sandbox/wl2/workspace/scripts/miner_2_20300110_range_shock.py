import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,2600)
 if x is None or len(x)<100: x=get_index_daily_data(s,2600)
 if x is not None and len(x): D[s]=x.set_index('date')['close'].astype(float)
px=pd.DataFrame(D).sort_index().ffill(); r=px.pct_change()
# Range-normalized one-day shock reversal: reverse the latest return relative to its 20d volatility.
# Signal is lagged one completed day before forward-return measurement.
f=(-r/(r.rolling(20,min_periods=10).std())).shift(1)
rows=[]
for i in range(len(px)-10):
 for h in [1,3,5,10]:
  z=pd.concat([f.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: rows.append((px.index[i],h,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
a=pd.DataFrame(rows,columns=['date','h','n','ic']).dropna()
for h in [1,3,5,10]:
 z=a[a.h==h].ic; print('h',h,'dates',len(z),'avgN',a[a.h==h].n.mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1)*np.sqrt(len(z)),'hit',(z>0).mean())
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'last',px.index[-1])
for name,mask in [('2020-22',a.date<'2023-01-01'),('2023-25',(a.date>='2023-01-01')&(a.date<'2026-01-01')),('2026-29',a.date>='2026-01-01')]:
 z=a[(a.h==1)&mask].ic; print(name,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1)*np.sqrt(len(z)))
f.tail(1).T.to_csv('scripts/miner_2_20300110_range_shock_signal.csv',header=False)
