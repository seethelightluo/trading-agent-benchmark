import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=1900)
 if d is not None and len(d): D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); v=r.rolling(20).std()
# volatility-normalized short trend, with cross-sectional rank; signal lag implicit via forward returns
f=(p.pct_change(5)/v.replace(0,np.nan)).rank(axis=1,pct=True)
for h in [1,3,5,10]:
 fr=p.pct_change(h).shift(-h); a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q):a.append(q);ns.append(len(z))
 x=pd.Series(a); print('h',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
print('assets',len(D),'rows',len(p),'coverage',round(f.notna().sum(axis=1).mean()/15,4),'turnover',round(f.diff().abs().mean(axis=1).mean(),4))
for a,b in [(2020,2021),(2022,2023),(2024,2025),(2026,2027)]:
 q=[]
 for dt in f.index:
  if a<=dt.year<=b:
   z=pd.concat([f.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
   if len(z)>=8:
    c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
    if np.isfinite(c):q.append(c)
 print('regime',a,b,'dates',len(q),'IC',round(np.mean(q),6))
