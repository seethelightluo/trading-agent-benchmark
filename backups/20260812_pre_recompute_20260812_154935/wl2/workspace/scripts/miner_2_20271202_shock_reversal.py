import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,2200)
 if x is None or len(x)<80:x=get_index_daily_data(s,2200)
 if x is not None and len(x):D[s]=x.set_index('date')['close'].astype(float)
px=pd.DataFrame(D).sort_index().ffill(); r=px.pct_change()
# causal short-horizon shock reversal, normalized by recent volatility
f=(-r.rolling(3,min_periods=3).sum()/ (r.rolling(20,min_periods=10).std()*np.sqrt(3))).shift(1)
rows=[]
for i in range(len(px)-10):
 for h in [1,3,5,10]:
  z=pd.concat([f.iloc[i],(px.iloc[i+h]/px.iloc[i]-1)],axis=1).dropna()
  if len(z)>=8: rows.append((px.index[i],h,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
a=pd.DataFrame(rows,columns=['date','h','n','ic']).dropna()
for h in [1,3,5,10]:
 z=a[a.h==h].ic; print('h',h,'dates',len(z),'avgN',a[a.h==h].n.mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1)*np.sqrt(len(z)),'hit',(z>0).mean())
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for name,mask in [('2020-22',a.date<'2023-01-01'),('2023-25',(a.date>='2023-01-01')&(a.date<'2026-01-01')),('2026-27',a.date>='2026-01-01')]:
 z=a[(a.h==1)&mask].ic;print(name,len(z),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(len(z)))
print('last',px.index[-1])
